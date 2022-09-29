# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Visualization routines using matplotlib
"""
import copy
import glob
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.patches import Ellipse, RegularPolygon, Rectangle, Circle

from enum import Enum, unique
from astropy.coordinates import Angle, SkyCoord
from numpy import cos, sin
from astropy.coordinates import BaseCoordinateFrame

__all__ = ["CameraDisplay"]


def polar_to_cart(rho, phi):
    """"returns r, theta(degrees)"""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def rotation_matrix_2d(angle):
    """construct a 2D rotation matrix as a numpy NDArray that rotates a
    vector clockwise. Angle should be any object that can be converted
    into an `astropy.coordinates.Angle`
    """
    psi = Angle(angle).rad
    return np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])

class PixelShape(Enum):
    """Supported Pixel Shapes Enum"""

    CIRCLE = "circle"
    SQUARE = "square"
    HEXAGON = "hexagon"

    @classmethod
    def from_string(cls, name):
        """
        Convert a string represenation to the enum value
        This function supports abbreviations and for backwards compatibility
        "rect" as alias for "square".
        """
        name = name.lower()

        if name.startswith("hex"):
            return cls.HEXAGON

        if name.startswith("rect") or name == "square":
            return cls.SQUARE

        if name.startswith("circ"):
            return cls.CIRCLE

        raise TypeError(f"Unknown pixel shape {name}")


#: mapper from simtel pixel shape integers to our shape and rotation angle
SIMTEL_PIXEL_SHAPES = {
    0: (PixelShape.CIRCLE, Angle(0, u.deg)),
    1: (PixelShape.HEXAGON, Angle(0, u.deg)),
    2: (PixelShape.SQUARE, Angle(0, u.deg)),
    3: (PixelShape.HEXAGON, Angle(30, u.deg)),
}

class CameraGeometry:
    """`CameraGeometry` is a class that stores information about a
    Cherenkov Camera that us useful for imaging algorithms and
    displays. It contains lists of pixel positions, areas, pixel
    shapes, as well as a neighbor (adjacency) list and matrix for each pixel.
    In general the neighbor_matrix attribute should be used in any algorithm
    needing pixel neighbors, since it is much faster. See for example
    `ctapipe.image.tailcuts_clean`
    The class is intended to be generic, and work with any Cherenkov
    Camera geometry, including those that have square vs hexagonal
    pixels, gaps between pixels, etc.
    Parameters
    ----------
    self: type
        description
    camera_name: str
         Camera name (e.g. NectarCam, LSTCam, ...)
    pix_id: array(int)
        pixels id numbers
    pix_x: array with units
        position of each pixel (x-coordinate)
    pix_y: array with units
        position of each pixel (y-coordinate)
    pix_area: array(float)
        surface area of each pixel, if None will be calculated
    neighbors: list(arrays)
        adjacency list for each pixel
    pix_type: string
        either 'rectangular' or 'hexagonal'
    pix_rotation: value convertable to an `astropy.coordinates.Angle`
        rotation angle with unit (e.g. 12 * u.deg), or "12d"
    cam_rotation: overall camera rotation with units
    """

    _geometry_cache = {}  # dictionary CameraGeometry instances for speed

    def __init__(
        self,
        camera_name,
        pix_id,
        pix_x,
        pix_y,
        pix_area,
        pix_type,
        pix_rotation="0d",
        cam_rotation="0d",
        neighbors=None,
        apply_derotation=True,
        frame=None,
    ):
        if pix_x.ndim != 1 or pix_y.ndim != 1:
            raise ValueError(
                f"Pixel coordinates must be 1 dimensional, got {pix_x.ndim}"
            )

        assert len(pix_x) == len(pix_y), "pix_x and pix_y must have same length"

        if isinstance(pix_type, str):
            pix_type = PixelShape.from_string(pix_type)
        elif not isinstance(pix_type, PixelShape):
            raise TypeError(
                f"pix_type most be a PixelShape or the name of a PixelShape, got {pix_type}"
            )

        self.n_pixels = len(pix_x)
        self.camera_name = camera_name
        self.pix_id = pix_id
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.pix_area = pix_area
        self.pix_type = pix_type

        if not isinstance(pix_rotation, Angle):
            pix_rotation = Angle(pix_rotation)

        if not isinstance(cam_rotation, Angle):
            cam_rotation = Angle(cam_rotation)

        self.pix_rotation = pix_rotation
        self.cam_rotation = cam_rotation

        self._neighbors = neighbors
        self.frame = frame

        if neighbors is not None:
            if isinstance(neighbors, list):
                lil = lil_matrix((self.n_pixels, self.n_pixels), dtype=bool)
                for pix_id, neighbors in enumerate(neighbors):
                    lil[pix_id, neighbors] = True
                self._neighbors = lil.tocsr()
            else:
                self._neighbors = csr_matrix(neighbors)

        if self.pix_area is None:
            self.pix_area = self.guess_pixel_area(pix_x, pix_y, pix_type)

        if apply_derotation:
            self.rotate(self.cam_rotation)

            # cache border pixel mask per instance
        self.border_cache = {}

    @classmethod
    def guess_pixel_area(cls, pix_x, pix_y, pix_type):
        """
        Guess pixel area based on the pixel type and layout.
        This first uses `guess_pixel_width` and then calculates
        area from the given pixel type.
        Note this will not work on cameras with varying pixel sizes.
        """

        dist = cls.guess_pixel_width(pix_x, pix_y)

        if pix_type == PixelShape.HEXAGON:
            area = 2 * np.sqrt(3) * (dist / 2) ** 2
        elif pix_type == PixelShape.SQUARE:
            area = dist ** 2
        else:
            raise KeyError("unsupported pixel type")

        return np.ones(pix_x.shape) * area

    @staticmethod
    def guess_pixel_width(pix_x, pix_y):
        """
        Calculate pixel diameter by looking at the minimum distance between pixels
        Note this will not work on cameras with varying pixel sizes or gaps
        Returns
        -------
            in-circle diameter for hexagons, edge width for square pixels
        """
        return np.min(
            np.sqrt((pix_x[1:] - pix_x[0]) ** 2 + (pix_y[1:] - pix_y[0]) ** 2)
        )

    def pixel_width(self):
            """
            in-circle diameter for hexagons, edge width for square pixels,
            diameter for circles.
            This is calculated from the pixel area.
            """

            if self.pix_type == PixelShape.HEXAGON:
                width = 2 * np.sqrt(self.pix_area / (2 * np.sqrt(3)))
            elif self.pix_type == PixelShape.SQUARE:
                width = np.sqrt(self.pix_area)
            elif self.pix_type == PixelShape.CIRCLE:
                width = 2 * np.sqrt(self.pix_area / np.pi)
            else:
                raise NotImplementedError(
                    f"Cannot calculate pixel width for type {self.pix_type!r}"
                )
            return width

    def rotate(self, angle):
        """rotate the camera coordinates about the center of the camera by
        specified angle. Modifies the CameraGeometry in-place (so
        after this is called, the pix_x and pix_y arrays are
        rotated.
        Notes
        -----
        This is intended only to correct simulated data that are
        rotated by a fixed angle.  For the more general case of
        correction for camera pointing errors (rotations,
        translations, skews, etc), you should use a true coordinate
        transformation defined in `ctapipe.coordinates`.
        Parameters
        ----------
        angle: value convertable to an `astropy.coordinates.Angle`
            rotation angle with unit (e.g. 12 * u.deg), or "12d"
        """
        angle = Angle(angle)
        rotmat = rotation_matrix_2d(angle)
        rotated = np.dot(rotmat.T, [self.pix_x.value, self.pix_y.value])
        self.pix_x = rotated[0] * self.pix_x.unit
        self.pix_y = rotated[1] * self.pix_x.unit

        # do not use -=, copy is intentional here
        self.pix_rotation = self.pix_rotation - angle
        self.cam_rotation = Angle(0, unit=u.deg)

class CameraDisplay:
    """
    Camera Display using matplotlib.
    Parameters
    ----------
    geometry : `~ctapipe.instrument.CameraGeometry`
        Definition of the Camera/Image
    image: array_like
        array of values corresponding to the pixels in the CameraGeometry.
    ax : `matplotlib.axes.Axes`
        A matplotlib axes object to plot on, or None to create a new one
    title : str (default "Camera")
        Title to put on camera plot
    norm : str or `matplotlib.colors.Normalize` instance (default 'lin')
        Normalization for the color scale.
        Supported str arguments are
        - 'lin': linear scale
        - 'log': logarithmic scale (base 10)
    cmap : str or `matplotlib.colors.Colormap` (default 'hot')
        Color map to use (see `matplotlib.cm`)
    allow_pick : bool (default False)
        if True, allow user to click and select a pixel
    autoupdate : bool (default True)
        redraw automatically (otherwise need to call plt.draw())
    autoscale : bool (default True)
        rescale the vmin/vmax values when the image changes.
        This is set to False if ``set_limits_*`` is called to explicity
        set data limits.
    Notes
    -----
    Speed:
        CameraDisplay is not intended to be very fast (matplotlib
        is not a very speed performant graphics library, it is
        intended for nice output plots). However, most of the
        slowness of CameraDisplay is in the constructor.  Once one is
        displayed, changing the image that is displayed is relatively
        fast and efficient. Therefore it is best to initialize an
        instance, and change the data, rather than generating new
        CameraDisplays.
    Pixel Implementation:
        Pixels are rendered as a
        `matplotlib.collections.PatchCollection` of Polygons (either 6
        or 4 sided).  You can access the PatchCollection directly (to
        e.g. change low-level style parameters) via
        ``CameraDisplay.pixels``
    Output:
        Since CameraDisplay uses matplotlib, any display can be
        saved to any output file supported via
        plt.savefig(filename). This includes ``.pdf`` and ``.png``.
    """

    def __init__(
        self,
        geometry,
        image=None,
        ax=None,
        title=None,
        norm="lin",
        cmap=None,
        allow_pick=False,
        autoupdate=True,
        autoscale=True,
        show_frame=True,
    ):
        self.axes = ax if ax is not None else plt.gca()
        self.pixels = None
        self.colorbar = None
        self.autoupdate = autoupdate
        self.autoscale = autoscale
        self._active_pixel = None
        self._active_pixel_label = None
        self._axes_overlays = []
        #plt.figure(figsize=(25,20), dpi=80)
        self.geom = geometry

        if title is None:
            title = f"{geometry.camera_name}"

        # initialize the plot and generate the pixels as a
        # RegularPolyCollection

        patches = []

        if hasattr(self.geom, "mask"):
            self.mask = self.geom.mask
        else:
            self.mask = np.ones_like(self.geom.pix_x.value, dtype=bool)

        pix_x = self.geom.pix_x.value[self.mask]
        pix_y = self.geom.pix_y.value[self.mask]
        pix_width = self.geom.pixel_width().value[self.mask]

        for x, y, w in zip(pix_x, pix_y, pix_width):
            if self.geom.pix_type == PixelShape.HEXAGON:
                r = w / np.sqrt(3)
                patch = RegularPolygon(
                    (x, y),
                    6,
                    radius=r,
                    orientation=self.geom.pix_rotation.to_value(u.rad),
                    fill=True,
                )
            elif self.geom.pix_type == PixelShape.CIRCLE:
                patch = Circle((x, y), radius=w / 2, fill=True)
            elif self.geom.pix_type == PixelShape.SQUARE:
                patch = Rectangle(
                    (x - w / 2, y - w / 2),
                    width=w,
                    height=w,
                    angle=self.geom.pix_rotation.to_value(u.deg),
                    fill=True,
                )
            else:
                raise ValueError(f"Unsupported pixel_shape {self.geom.pix_type}")

            patches.append(patch)

        self.pixels = PatchCollection(patches, cmap=cmap, linewidth=0)
        self.axes.add_collection(self.pixels)

        self.pixel_highlighting = copy.copy(self.pixels)
        self.pixel_highlighting.set_facecolor("none")
        self.pixel_highlighting.set_linewidth(0)
        self.axes.add_collection(self.pixel_highlighting)

        # Set up some nice plot defaults

        self.axes.set_aspect("equal", "datalim")
        self.axes.set_title(title)
        self.axes.autoscale_view()

        if show_frame:
            self.add_frame_name()
        # set up a patch to display when a pixel is clicked (and
        # pixel_picker is enabled):

        self._active_pixel = copy.copy(patches[0])
        self._active_pixel.set_facecolor("r")
        self._active_pixel.set_alpha(0.5)
        self._active_pixel.set_linewidth(2.0)
        self._active_pixel.set_visible(False)
        self.axes.add_patch(self._active_pixel)

        if hasattr(self._active_pixel, "xy"):
            center = self._active_pixel.xy
        else:
            center = self._active_pixel.center

        self._active_pixel_label = self.axes.text(
            *center, "0", horizontalalignment="center", verticalalignment="center"
        )
        self._active_pixel_label.set_visible(False)

        # enable ability to click on pixel and do something (can be
        # enabled on-the-fly later as well:

        if allow_pick:
            self.enable_pixel_picker()

        if image is not None:
            self.image = image
        else:
            self.image = np.zeros_like(self.geom.pix_id, dtype=np.float64)

        self.norm = norm
        self.auto_set_axes_labels()

    def highlight_pixels(self, pixels, color="g", linewidth=1, alpha=0.75):
        """
        Highlight the given pixels with a colored line around them
        Parameters
        ----------
        pixels : index-like
            The pixels to highlight.
            Can either be a list or array of integers or a
            boolean mask of length number of pixels
        color: a matplotlib conform color
            the color for the pixel highlighting
        linewidth: float
            linewidth of the highlighting in points
        alpha: 0 <= alpha <= 1
            The transparency
        """

        l = np.zeros_like(self.image)
        l[pixels] = linewidth
        self.pixel_highlighting.set_linewidth(l)
        self.pixel_highlighting.set_alpha(alpha)
        self.pixel_highlighting.set_edgecolor(color)
        self._update()

    def enable_pixel_picker(self):
        """ enable ability to click on pixels """
        self.pixels.set_picker(True)
        self.pixels.set_pickradius(self.geom.pixel_width().value[0] / 2)
        self.axes.figure.canvas.mpl_connect("pick_event", self._on_pick)

    @property
    def norm(self):
        """
        The norm instance of the Display
        Possible values:
        - "lin": linear scale
        - "log": log scale (cannot have negative values)
        - "symlog": symmetric log scale (negative values are ok)
        -  any matplotlib.colors.Normalize instance, e. g. PowerNorm(gamma=-2)
        """
        return self.pixels.norm

    @norm.setter
    def norm(self, norm):

        if norm == "lin":
            self.pixels.norm = Normalize()
        elif norm == "log":
            self.pixels.norm = LogNorm()
            self.pixels.autoscale()  # this is to handle matplotlib bug #5424
        elif norm == "symlog":
            self.pixels.norm = SymLogNorm(linthresh=1.0, base=10)
            self.pixels.autoscale()
        elif isinstance(norm, Normalize):
            self.pixels.norm = norm
        else:
            raise ValueError(
                "Unsupported norm: '{}', options are 'lin',"
                "'log','symlog', or a matplotlib Normalize object".format(norm)
            )

        self.update(force=True)
        self.pixels.autoscale()

    @property
    def cmap(self):
        """
        Color map to use. Either name or `matplotlib.colors.Colormap`
        """
        return self.pixels.get_cmap()

    @cmap.setter
    def cmap(self, cmap):
        self.pixels.set_cmap(cmap)
        self._update()

    @property
    def image(self):
        """The image displayed on the camera (1D array of pixel values)"""
        return self.pixels.get_array()

    @image.setter
    def image(self, image):
        """
        Change the image displayed on the Camera.
        Parameters
        ----------
        image: array_like
            array of values corresponding to the pixels in the CameraGeometry.
        """
        image = np.asanyarray(image)
        if image.shape != self.geom.pix_x.shape:
            raise ValueError(
                (
                    "Image has a different shape {} than the " "given CameraGeometry {}"
                ).format(image.shape, self.geom.pix_x.shape)
            )

        self.pixels.set_array(np.ma.masked_invalid(image[self.mask]))
        self.pixels.changed()
        if self.autoscale:
            self.pixels.autoscale()
        self._update()

    def _update(self, force=False):
        """ signal a redraw if autoupdate is turned on """
        if self.autoupdate:
            self.update(force)

    def update(self, force=False):
        """ redraw the display now """
        self.axes.figure.canvas.draw()
        if self.colorbar is not None:
            if force is True:
                self.colorbar.update_bruteforce(self.pixels)
            else:
                self.colorbar.update_normal(self.pixels)
            self.colorbar.draw_all()

    def add_colorbar(self, **kwargs):
        """
        add a colorbar to the camera plot
        kwargs are passed to ``figure.colorbar(self.pixels, **kwargs)``
        See matplotlib documentation for the supported kwargs:
        http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.colorbar
        """
        if self.colorbar is not None:
            raise ValueError(
                "There is already a colorbar attached to this CameraDisplay"
            )
        else:
            if "ax" not in kwargs:
                kwargs["ax"] = self.axes
            self.colorbar = self.axes.figure.colorbar(self.pixels, **kwargs)
        self.update()

    def add_frame_name(self, color="grey"):
            """ label the frame type of the display (e.g. CameraFrame) """

            frame_name = (
                self.geom.frame.__class__.__name__
                if self.geom.frame is not None
                else "Unknown Frame"
            )
            self.axes.text(  # position text relative to Axes
                1.0,
                0.0,
                frame_name,
                ha="right",
                va="bottom",
                transform=self.axes.transAxes,
                color=color,
                fontsize="smaller",
            )

    def auto_set_axes_labels(self):
            """ set the axes labels based on the Frame attribute"""
            axes_labels = ("X", "Y")
            if self.geom.frame is not None:
                axes_labels = list(
                    self.geom.frame.get_representation_component_names().keys()
                )

            self.axes.set_xlabel(f"{axes_labels[0]}  ({self.geom.pix_x.unit})")
            self.axes.set_ylabel(f"{axes_labels[1]}  ({self.geom.pix_y.unit})")

    def add_ellipse(self, centroid, length, width, angle, asymmetry=0.0, **kwargs):
        """
        plot an ellipse on top of the camera

        Parameters
        ----------
        centroid: (float, float)
            position of centroid
        length: float
            major axis
        width: float
            minor axis
        angle: float
            rotation angle wrt x-axis about the centroid, anticlockwise, in radians
        asymmetry: float
            3rd-order moment for directionality if known
        kwargs:
            any MatPlotLib style arguments to pass to the Ellipse patch

        """
        ellipse = Ellipse(
            xy=centroid,
            width=length,
            height=width,
            angle=np.degrees(angle),
            **kwargs,
            label = 'Хилласова аппроксимация изображения'
        )
        self.axes.add_patch(ellipse)
        self.update()
        return ellipse

    def add_source(self, xy_source, zorder):
        source = Circle(xy_source, radius=1.0, color='r', label = 'Положение источника', zorder=10)
        self.axes.add_patch(source)
        self.update()
        self._axes_overlays.append(source)


    def overlay_moments(
        self, hillas_parameters, with_label=False, keep_old=False, **kwargs
    ):
        """helper to overlay ellipse from a `~ctapipe.containers.HillasParametersContainer` structure

        Parameters
        ----------
        hillas_parameters: `HillasParametersContainer`
            structuring containing Hillas-style parameterization
        with_label: bool
            If True, show coordinates of centroid and width and length
        keep_old: bool
            If True, to not remove old overlays
        kwargs: key=value
            any style keywords to pass to matplotlib (e.g. color='red'
            or linewidth=6)
        """
        if not keep_old:
            self.clear_overlays()

        # strip off any units
        cen_x = hillas_parameters[0]/0.1206
        cen_y = hillas_parameters[1]/0.1206
        length = hillas_parameters[3]/0.1206
        width = hillas_parameters[2]/0.1206

        el = self.add_ellipse(
            centroid=(cen_x, cen_y),
            length=length * 2,
            width=width * 2,
            angle=hillas_parameters[4],
            **kwargs,
        )
        #print(np.degrees(hillas_parameters[4]))

        self._axes_overlays.append(el)

        if with_label:
            text = self.axes.text(
                cen_x,
                cen_y,
                "({:.02f},{:.02f})\n[w={:.02f},l={:.02f}]".format(
                    cen_x,
                    cen_y,
                    width,
                    length,
                ),
                color=el.get_edgecolor(),
            )

            self._axes_overlays.append(text)


    def clear_overlays(self):
        """Remove added overlays from the axes"""
        while self._axes_overlays:
            overlay = self._axes_overlays.pop()
            overlay.remove()

    def _on_pick(self, event):
        """ handler for when a pixel is clicked """
        pix_id = event.ind[-1]
        x = self.geom.pix_x[pix_id].value
        y = self.geom.pix_y[pix_id].value

        if self.geom.pix_type in (PixelShape.HEXAGON, PixelShape.CIRCLE):
            self._active_pixel.xy = (x, y)
        else:
            w = self.geom.pixel_width.value[0]
            self._active_pixel.xy = (x - w / 2.0, y - w / 2.0)

        self._active_pixel.set_visible(True)
        self._active_pixel_label.set_x(x)
        self._active_pixel_label.set_y(y)
        self._active_pixel_label.set_text(f"{pix_id:003d}")
        self._active_pixel_label.set_visible(True)
        self._update()
        self.on_pixel_clicked(pix_id)  # call user-function

    def on_pixel_clicked(self, pix_id):
        """virtual function to overide in sub-classes to do something special
        when a pixel is clicked
        """
        print(f"Clicked pixel_id {pix_id}")

    def show(self):
            self.axes.figure.show()

    def add_frame_name(self, color="grey"):
        """ label the frame type of the display (e.g. CameraFrame) """

        frame_name = ("TAIGA-IACT FoV ")
        self.axes.text(  # position text relative to Axes
            1.0,
            0.0,
            frame_name,
            ha="right",
            va="bottom",
            transform=self.axes.transAxes,
            color=color,
            fontsize="smaller",
        )

def plot_source_location(hillas, xy_source, source_reco):
    cluster, unused1, unused2, p_x, p_y, hig_ch, low_ch, pix_maroc_id = np.loadtxt('/hdd/IACTs/Corsika/readbin/xy_iact02_2020jul.txt', unpack=True)
    dict_id = {}
    for it in range(len(cluster)):
        dict_id[(int(cluster[it]), int(hig_ch[it]))] = it
    pix_contin_id = np.arange(len(pix_maroc_id))
    p_x = p_x * u.cm
    p_y = p_y * u.cm
    geom_iact = CameraGeometry(camera_name = ' ', pix_id = pix_contin_id, pix_x = p_x, pix_y = p_y, pix_area = None, pix_rotation="22.5d", cam_rotation="-37.5d", pix_type = 'hex') 
    camera_image = CameraDisplay(geometry = geom_iact)
    camera_image.add_source(xy_source, zorder=10)
    for hillas_i in hillas:
        camera_image.overlay_moments(hillas_i, keep_old=True, color='White', fill=0)  
        length_line = 0
        if(hillas_i[4] > 0):
            length_line = 30.0
        else:
            length_line = -30.0
        endy = hillas_i[1]/0.1206 + length_line * np.sin(hillas_i[4])
        endx = hillas_i[0]/0.1206 + length_line * np.cos(hillas_i[4])
        print(hillas_i[0]/0.1206, hillas_i[1]/0.1206, endx, endy)
        plt.plot([hillas_i[0]/0.1206, endx], [hillas_i[1]/0.1206, endy], color='orange', zorder=5)
        camera_image.overlay_moments(source_reco, keep_old=True, color='g', fill=1, zorder=15)
    plt.savefig('/hdd/IACTs/Corsika/readbin/data/new_cone/1km/Crab/for_paper/camera_FoV.png', dpi = 400, facecolor="w")
    plt.show()
    

def plot_image(i_tel, pix_array, hillas, xy_source):
    cam_coord_path = ['/hdd/IACTs/Corsika/readbin/xy_turn_2019o.txt', '/hdd/IACTs/Corsika/readbin/xy_iact02_2020jul.txt', '/hdd/IACTs/Corsika/readbin/xy_iact02_2020jul.txt','/hdd/IACTs/Corsika/readbin/xy_iact02_2020jul.txt', '/hdd/IACTs/Corsika/readbin/xy_iact02_2020jul.txt']
    #cam_coord_IACT01 = '/hdd/IACTs/Corsika/readbin/xy_turn_2019o.txt' #поменять на путь где лежит файл с координатами телескопа
    #cam_coord_IACT02 = '/hdd/IACTs/Corsika/readbin/xy_iact02_2020jul.txt' #поменять на путь где лежит файл с координатами телескопа
    #cam_coord_IACT03 = '/hdd/IACTs/Corsika/readbin/xy_jan22.i3' #поменять на путь где лежит файл с координатами телескопа
    cluster, unused1, unused2, p_x, p_y, hig_ch, low_ch, pix_maroc_id = np.loadtxt(cam_coord_path[i_tel], unpack=True) #cam_coord_IACT02 поменять в зависимости от нужного телескопа
    dict_id = {}
    for it in range(len(cluster)):
        dict_id[(int(cluster[it]), int(hig_ch[it]))] = it

    pix_contin_id = np.arange(len(pix_maroc_id))
    p_x = p_x * u.cm
    p_y = p_y * u.cm
    #px_rot = p_x * cos(-37.5*np.pi/180) + p_y * sin(-37.5*np.pi/180)
    #py_rot = p_y * cos(-37.5*np.pi/180) - p_x * sin(-37.5*np.pi/180)
    #print(p_x, p_y, pix_contin_id)
    geom_iact = CameraGeometry(camera_name = 'TAIGA-IACT0' + str(i_tel+1), pix_id = pix_contin_id, pix_x = p_x, pix_y = p_y, pix_area = None, pix_rotation="22.5d", cam_rotation="-37.5d", pix_type = 'hex') #поменять номер телескопа на нужный в TAIGA-IACT02
    #print(geom_iact.pixel_width().value)
    #out_list = glob.glob('/hdd/IACTs/Camera3/090921.13/outs/*.out_*')
    camera_image = CameraDisplay(geometry = geom_iact)
    camera_image.add_colorbar()
    camera_image.overlay_moments(hillas, color='White', fill=0)
    camera_image.add_source(xy_source, zorder=10)
    amp_ar = []
    for i in range(len(pix_contin_id)):
        amp_ar.append(0)
    for pix in pix_array:
        if (int(pix[0]), int(pix[1])) in dict_id.keys():
            amp_ar[dict_id[int(pix[0]), int(pix[1])]] = pix[2]
                #for b in file.readline().split():
                    #print(float(b))
                    #if(float(b) > 0.0):
            #print(amp_ar)
    camera_image.image = amp_ar
    camera_image.enable_pixel_picker()
    plt.savefig('/hdd/IACTs/Corsika/readbin/data/new_cone/1km/Crab/for_paper/image_%d.png' % i_tel, dpi = 400, facecolor="w")
        #if event_number == 155241:
        #plt.savefig('/hdd/IACTs/Corsika/readbin/data/old_cone/bpe607_31_da0.1_md5/%d_%d_IACT0' + str(i_tel+1) + '.png' % (portion[0]+1, int(event_number))) #поменять путь куда сохранить картинку
    #plt.legend()
    plt.show()
#print(dict_id)
#plt.savefig('/hdd/IACTs/PyTest/cta/example.png')
