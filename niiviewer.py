#!/usr/bin/env python3
"""NIFTI Dataset Viewer.
View images with bboxes from the NIFTI dataset.
edit line 77 if the orientation of the image does not work fine
"""
import argparse
import colorsys
import json
import nibabel as nib
import logging
import os
import random
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from skimage.measure import label, regionprops
from turtle import __forwardmethods

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser(
    description="View images with bboxes from the NIFTI dataset")
parser.add_argument(
    "-i",
    "--images",
    default="",
    type=str,
    metavar="PATH",
    required=True,
    help="Path to the folder containing original images")
parser.add_argument(
    "-a",
    "--annotations",
    default="",
    type=str,
    metavar="PATH",
    required=True,
    help="Path to the folder containing annotations",
)
parser.add_argument(
    "-c",
    "--categories",
    default="",
    type=str,
    metavar="PATH",
    required=True,
    help="Path to the JSON file with category information",
)


class Data:
    """Handles data related stuff."""

    def __init__(self, image_dir, annotations_dir, config_file):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        image_files = [os.path.join(self.image_dir, f)
                       for f in os.listdir(self.image_dir)]
        # NOTE: image list is based on annotations file
        self.images = ImageList(image_files)
        self.categories = get_categories(config_file)  # Dataset categories
        self.current_layer = 0  # Current layer
        # Prepare the very first image
        self.next_image()
        self.load_array()
        self.max_layer = self.image_array.shape[-1] - 1

    def normalize(self, array):
        """Normalizes the array."""
        return ((array - np.min(array)) / (np.max(array) - np.min(array))*255).astype(np.uint8)

    def prepare_array(self, object_based_coloring: bool = False):
        """Prepares image and annotation arrays for the current layer."""
        image_current_layer_array = np.flip(np.rot90(self.normalize(
            self.image_array[..., self.current_layer]), k=1), axis=0)
        annotation_current_layer_array = self.annotation_array[...,
                                                               self.current_layer]
        labeled_array = label(annotation_current_layer_array)
        regions = regionprops(labeled_array)
        names_colors = []
        # Get new colors for the image if object based coloring is on
        obj_colors = prepare_colors(len(regions))
        # Update name-color pairs using list comprehension
        names_colors = [
            [self.categories[annotation_current_layer_array[region.coords[0][0], region.coords[0][1]]][0],
             obj_colors[i] if object_based_coloring else self.categories[annotation_current_layer_array[region.coords[0][0], region.coords[0][1]]][1]]
            for i, region in enumerate(regions)
        ]
        current_layer_object_categories = [int(
            annotation_current_layer_array[region.coords[0][0], region.coords[0][1]]) for region in regions]
        current_layer_categories = sorted(
            list(set(current_layer_object_categories)))
        return image_current_layer_array, annotation_current_layer_array, regions, names_colors, current_layer_object_categories, current_layer_categories

    def next_image(self):
        """Loads the next image in a list."""
        self.current_image = self.images.next()
        self.current_annotation = self.current_image.replace(
            self.image_dir, self.annotations_dir)
        self.load_array()
        self.max_layer = self.image_array.shape[-1] - 1

    def previous_image(self):
        """Loads the previous image in a list."""
        self.current_image = self.images.prev()
        self.current_annotation = self.current_image.replace(
            self.image_dir, self.annotations_dir)
        self.load_array()
        self.max_layer = self.image_array.shape[-1] - 1

    def load_array(self):
        """Loads array from the NIFTI dataset."""
        self.image_array = nib.load(self.current_image).get_fdata()
        self.annotation_array = nib.load(self.current_annotation).get_fdata()


def prepare_layers(image_current_layer_array: np.array):
    """Opens image, creates draw context."""
    # Open image
    img_open = Image.fromarray(
        image_current_layer_array, mode="L").convert("RGBA")
    # Create layer for bboxes and masks
    draw_layer = Image.new("RGBA", img_open.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(draw_layer)
    return img_open, draw_layer, draw


def prepare_colors(n_objects: int, shuffle: bool = False) -> list:
    """Get some colors."""
    # Get some colors
    hsv_tuples = [(x / n_objects, 1.0, 1.0) for x in range(n_objects)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    # Shuffle colors
    if shuffle:
        random.seed(417)
        random.shuffle(colors)
        random.seed(None)
    return colors


def get_categories(config_file: str) -> dict:
    """Extracts categories from config file and prepares color for each one."""
    # Parse categories
    with open(config_file, 'r') as f:
        categories = json.load(f)["categories"]
    colors = prepare_colors(n_objects=len(categories), shuffle=False)
    categories_list = list(
        zip(
            [[category["id"], category["name"]] for category in categories],
            colors,
        )
    )
    categories = dict([[cat[0][0], [cat[0][1], cat[1]]]
                      for cat in categories_list])
    return categories


def draw_bboxes(draw, regions, labels_on, names_colors, ignore, width, label_size):
    """Puts rectangles on the image."""
    # Extracting bbox coordinates from regions from regionprops
    bboxes = [region.bbox for region in regions]
    # Draw bboxes
    for i, (c, b) in enumerate(zip(names_colors, bboxes)):
        if ignore and ignore[i]:
            continue
        draw.rectangle(b, outline=c[-1], width=width)
        if labels_on:
            text = c[0]
            try:
                try:
                    # Should work for Linux
                    font = ImageFont.truetype(
                        "DejaVuSans.ttf", size=label_size)
                except OSError:
                    # Should work for Windows
                    font = ImageFont.truetype("Arial.ttf", size=label_size)
            except OSError:
                # Load default, note no resize option
                # TODO: Implement notification message as popup window
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx0 = b[0]
            ty0 = b[1] - th
            # TODO: Looks weird! We need image dims to make it right
            tx0 = max(b[0], max(b[0], tx0)) if tx0 < 0 else tx0
            ty0 = max(b[1], max(0, ty0)) if ty0 < 0 else ty0
            tx1 = tx0 + tw
            ty1 = ty0 + th
            # TODO: The same here
            if tx1 > b[2]:
                tx0 = max(0, tx0 - (tx1 - b[2]))
                tx1 = tw if tx0 == 0 else b[2]
            draw.rectangle((tx0, ty0, tx1, ty1), fill=c[-1])
            draw.text((tx0, ty0), text, (255, 255, 255), font=font)


def draw_masks(draw, regions, names_colors, ignore, alpha):
    """Draw masks over image."""
    # Draw masks
    for i, (c, r) in enumerate(zip(names_colors, regions)):
        if ignore and ignore[i]:
            continue
        fill = tuple(list(c[-1]) + [alpha])
        for coord in r.coords:
            draw.point((coord[0], coord[1]), fill)


class ImageList:
    """Handles iterating through the images."""

    def __init__(self, images: list):
        self.image_list = images or []
        self.n = -1
        self.max = len(self.image_list)

    def next(self):
        """Sets the next image as current."""
        self.n += 1
        if self.n < self.max:
            current_image = self.image_list[self.n]
        else:
            self.n = 0
            current_image = self.image_list[self.n]
        return current_image

    def prev(self):
        """Sets the previous image as current."""
        if self.n == 0:
            self.n = self.max - 1
            current_image = self.image_list[self.n]
        else:
            self.n -= 1
            current_image = self.image_list[self.n]
        return current_image


class ImagePanel(ttk.Frame):
    """ttk port of original turtle.ScrolledCanvas code."""

    def __init__(self, parent, width=768, height=480, canvwidth=600, canvheight=600):
        super().__init__(parent, width=width, height=height)
        self._rootwindow = self.winfo_toplevel()
        self.width, self.height = width, height
        self.canvwidth, self.canvheight = canvwidth, canvheight
        self.bg = "gray15"
        self.pack(fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg=self.bg,
            relief="sunken",
            borderwidth=2,
        )
        self.hscroll = ttk.Scrollbar(
            parent, command=self._canvas.xview, orient=tk.HORIZONTAL)
        self.vscroll = ttk.Scrollbar(parent, command=self._canvas.yview)
        self._canvas.configure(
            xscrollcommand=self.hscroll.set, yscrollcommand=self.vscroll.set)
        self.rowconfigure(0, weight=1, minsize=0)
        self.columnconfigure(0, weight=1, minsize=0)
        self._canvas.grid(
            padx=1,
            in_=self,
            pady=1,
            row=0,
            column=0,
            rowspan=1,
            columnspan=1,
            sticky=tk.NSEW,
        )
        self.vscroll.grid(
            padx=1,
            in_=self,
            pady=1,
            row=0,
            column=1,
            rowspan=1,
            columnspan=1,
            sticky=tk.NSEW,
        )
        self.hscroll.grid(
            padx=1,
            in_=self,
            pady=1,
            row=1,
            column=0,
            rowspan=1,
            columnspan=1,
            sticky=tk.NSEW,
        )
        self.reset()
        self._rootwindow.bind("<Configure>", self.on_resize)

    def reset(self, canvwidth=None, canvheight=None, bg=None):
        """Adjusts canvas and scrollbars according to given canvas size."""
        if canvwidth:
            self.canvwidth = canvwidth
        if canvheight:
            self.canvheight = canvheight
        if bg:
            self.bg = bg
        self._canvas.config(
            bg=bg,
            scrollregion=(
                -self.canvwidth // 2,
                -self.canvheight // 2,
                self.canvwidth // 2,
                self.canvheight // 2,
            ),
        )
        self._canvas.xview_moveto(
            0.5 * (self.canvwidth - self.width + 30) / self.canvwidth)
        self._canvas.yview_moveto(
            0.5 * (self.canvheight - self.height + 30) / self.canvheight)
        self.adjust_scrolls()

    def adjust_scrolls(self):
        """Adjusts scrollbars according to window- and canvas-size."""
        cwidth = self._canvas.winfo_width()
        cheight = self._canvas.winfo_height()
        self._canvas.xview_moveto(
            0.5 * (self.canvwidth - cwidth) / self.canvwidth)
        self._canvas.yview_moveto(
            0.5 * (self.canvheight - cheight) / self.canvheight)
        if cwidth < self.canvwidth:
            self.hscroll.grid(
                padx=1,
                in_=self,
                pady=1,
                row=1,
                column=0,
                rowspan=1,
                columnspan=1,
                sticky=tk.NSEW,
            )
        else:
            self.hscroll.grid_forget()
        if cheight < self.canvheight:
            self.vscroll.grid(
                padx=1,
                in_=self,
                pady=1,
                row=0,
                column=1,
                rowspan=1,
                columnspan=1,
                sticky=tk.NSEW,
            )
        else:
            self.vscroll.grid_forget()

    def on_resize(self, event):
        self.adjust_scrolls()

    def bbox(self, *args):
        return self._canvas.bbox(*args)

    def cget(self, *args, **kwargs):
        return self._canvas.cget(*args, **kwargs)

    def config(self, *args, **kwargs):
        self._canvas.config(*args, **kwargs)

    def bind(self, *args, **kwargs):
        self._canvas.bind(*args, **kwargs)

    def unbind(self, *args, **kwargs):
        self._canvas.unbind(*args, **kwargs)

    def focus_force(self):
        self._canvas.focus_force()


__forwardmethods(ImagePanel, tk.Canvas, "_canvas")


class StatusBar(ttk.Frame):
    """Shows status line on the bottom."""

    def __init__(self, parent):
        super().__init__(parent)
        # self.configure(bd="gray75")
        self.pack(side=tk.BOTTOM, fill=tk.X)
        self.file_count = ttk.Label(self, borderwidth=5, background="gray75")
        self.file_count.pack(side=tk.RIGHT)
        self.layer = ttk.Label(self, borderwidth=5, background="gray75")
        self.layer.pack(side=tk.RIGHT)
        self.file_name = ttk.Label(self, borderwidth=5, background="gray75")
        self.file_name.pack(side=tk.LEFT)
        self.nobjects = ttk.Label(self, borderwidth=5, background="gray75")
        self.nobjects.pack(side=tk.LEFT)
        self.ncategories = ttk.Label(self, borderwidth=5, background="gray75")
        self.ncategories.pack(side=tk.LEFT)


class Menu(tk.Menu):
    def __init__(self, parent):
        super().__init__(parent)
        # Define menu structure
        self.file = self.file_menu()
        self.view = self.view_menu()

    def file_menu(self):
        """File Menu."""
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Save", accelerator="Ctrl+S")
        menu.add_separator()
        menu.add_command(label="Exit", accelerator="Ctrl+Q")
        self.add_cascade(label="File", menu=menu)
        return menu

    def view_menu(self):
        """View Menu."""
        menu = tk.Menu(self, tearoff=False)
        menu.add_checkbutton(label="BBoxes", onvalue=True, offvalue=False)
        menu.add_checkbutton(label="Labels", onvalue=True, offvalue=False)
        menu.add_checkbutton(label="Masks", onvalue=True, offvalue=False)
        self.add_cascade(label="View", menu=menu)
        menu.colormenu = tk.Menu(menu, tearoff=0)
        menu.colormenu.add_radiobutton(label="Categories", value=False)
        menu.colormenu.add_radiobutton(label="Objects", value=True)
        menu.add_cascade(label="Coloring", menu=menu.colormenu)
        return menu


class ObjectsPanel(ttk.PanedWindow):
    """Panels with listed objects and categories for the image."""

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(side=tk.RIGHT, fill=tk.Y)
        # Categories subpanel
        self.category_subpanel = ttk.Frame()
        ttk.Label(
            self.category_subpanel,
            text="categories",
            borderwidth=2,
            background="gray50",
        ).pack(side=tk.TOP, fill=tk.X)
        self.category_box = tk.Listbox(
            self.category_subpanel, selectmode=tk.EXTENDED, exportselection=0)
        self.category_box.pack(side=tk.TOP, fill=tk.Y, expand=True)
        self.add(self.category_subpanel)
        # Objects subpanel
        self.object_subpanel = ttk.Frame()
        ttk.Label(self.object_subpanel, text="objects", borderwidth=2,
                  background="gray50").pack(side=tk.TOP, fill=tk.X)
        self.object_box = tk.Listbox(
            self.object_subpanel, selectmode=tk.EXTENDED, exportselection=0)
        self.object_box.pack(side=tk.TOP, fill=tk.Y, expand=True)
        self.add(self.object_subpanel)


class SlidersBar(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(side=tk.BOTTOM, fill=tk.X)
        # Bbox thickness controller
        self.bbox_slider = tk.Scale(
            self, label="bbox", from_=0, to=25, tickinterval=5, orient=tk.HORIZONTAL)
        self.bbox_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Label text size controller
        self.label_slider = tk.Scale(
            self, label="label", from_=10, to=20, tickinterval=2, orient=tk.HORIZONTAL)
        self.label_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Mask transparency controller
        self.mask_slider = tk.Scale(
            self, label="mask", from_=0, to=255, tickinterval=40, orient=tk.HORIZONTAL)
        self.mask_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)


class Controller:
    def __init__(self, data, root, image_panel, statusbar, menu, objects_panel, sliders):
        self.data = data  # data layer
        self.root = root  # root window
        self.image_panel = image_panel  # image panel
        self.statusbar = statusbar  # statusbar on the bottom
        self.menu = menu  # main menu on the top
        self.objects_panel = objects_panel
        self.sliders = sliders
        # StatusBar Vars
        self.file_count_status = tk.StringVar()
        self.file_name_status = tk.StringVar()
        self.layer_status = tk.StringVar()
        self.nobjects_status = tk.StringVar()
        self.ncategories_status = tk.StringVar()
        self.statusbar.file_count.configure(
            textvariable=self.file_count_status)
        self.statusbar.file_name.configure(textvariable=self.file_name_status)
        self.statusbar.layer.configure(textvariable=self.layer_status)
        self.statusbar.nobjects.configure(textvariable=self.nobjects_status)
        self.statusbar.ncategories.configure(
            textvariable=self.ncategories_status)
        # Menu Vars
        self.bboxes_on_global = tk.BooleanVar()  # Toggles bboxes globally
        self.bboxes_on_global.set(True)
        self.labels_on_global = tk.BooleanVar()  # Toggles category labels
        self.labels_on_global.set(True)
        self.masks_on_global = tk.BooleanVar()  # Toggles masks globally
        self.masks_on_global.set(True)
        self.coloring_on_global = tk.BooleanVar()  # Toggles objects/categories coloring
        # False for categories (defaults), True for objects
        self.coloring_on_global.set(False)
        # Menu Configuration
        self.menu.file.entryconfigure("Save", command=self.save_image)
        self.menu.file.entryconfigure("Exit", command=self.exit)
        self.menu.view.entryconfigure(
            "BBoxes", variable=self.bboxes_on_global, command=self.menu_view_bboxes)
        self.menu.view.entryconfigure(
            "Labels", variable=self.labels_on_global, command=self.menu_view_labels)
        self.menu.view.entryconfigure(
            "Masks", variable=self.masks_on_global, command=self.menu_view_masks)
        self.menu.view.colormenu.entryconfigure(
            "Categories",
            variable=self.coloring_on_global,
            command=self.menu_view_coloring,
        )
        self.menu.view.colormenu.entryconfigure(
            "Objects", variable=self.coloring_on_global, command=self.menu_view_coloring
        )
        self.root.configure(menu=self.menu)
        # Init local setup (for the current (active) image)
        self.bboxes_on_local = self.bboxes_on_global.get()
        self.labels_on_local = self.labels_on_global.get()
        self.masks_on_local = self.masks_on_global.get()
        self.coloring_on_local = self.coloring_on_global.get()
        # Objects Panel stuff
        self.selected_cats_ids = None
        self.selected_objs_ids = None
        self.unselected_objs = None
        self.category_box_content = tk.StringVar()
        self.object_box_content = tk.StringVar()
        self.objects_panel.category_box.configure(
            listvariable=self.category_box_content)
        self.objects_panel.object_box.configure(
            listvariable=self.object_box_content)
        # Sliders Setup
        self.bbox_thickness = tk.IntVar()
        self.bbox_thickness.set(1)
        self.label_size = tk.IntVar()
        self.label_size.set(10)
        self.mask_alpha = tk.IntVar()
        self.mask_alpha.set(128)
        self.sliders.bbox_slider.configure(
            variable=self.bbox_thickness, command=lambda e: self.update_img())
        self.sliders.label_slider.configure(
            variable=self.label_size, command=lambda e: self.update_img())
        self.sliders.mask_slider.configure(
            variable=self.mask_alpha, command=lambda e: self.update_img())
        # Bind all events
        self.bind_events()
        # Compose the very first image
        self.current_composed_image = None
        self.current_layer_categories = None
        self.current_layer_object_categories = None
        self.update_img()

    def set_locals(self):
        self.bboxes_on_local = self.bboxes_on_global.get()
        self.labels_on_local = self.labels_on_global.get()
        self.masks_on_local = self.masks_on_global.get()
        self.coloring_on_local = self.coloring_on_global.get()
        # Update sliders
        self.update_sliders_state()

    def compose_image(
        self,
        image_current_layer_array,
        regions,
        names_colors,
        bboxes_on: bool = True,
        labels_on: bool = True,
        masks_on: bool = True,
        ignore: list = None,
        width: int = 1,
        alpha: int = 128,
        label_size: int = 15,
    ):
        ignore = ignore or []  # list of objects to ignore
        img_open, draw_layer, draw = prepare_layers(image_current_layer_array)
        # Draw masks
        if masks_on:
            draw_masks(draw, regions, names_colors, ignore, alpha)
        # Draw bounding boxes
        if bboxes_on:
            draw_bboxes(draw, regions, labels_on, names_colors,
                        ignore, width, label_size)
        elif labels_on:
            draw_bboxes(draw, regions, labels_on,
                        names_colors, ignore, 0, label_size)
        del draw
        # Resulting image
        self.current_composed_image = Image.alpha_composite(
            img_open, draw_layer)

    def update_img(self, local=True, width=None, alpha=None, label_size=None):
        """Triggers image composition and sets composed image as current."""
        bboxes_on = self.bboxes_on_local if local else self.bboxes_on_global.get()
        labels_on = self.labels_on_local if local else self.labels_on_global.get()
        masks_on = self.masks_on_local if local else self.masks_on_global.get()
        coloring = self.coloring_on_local if local else self.coloring_on_global.get()
        # Prepare image
        (
            image_current_layer_array,
            _,
            regions,
            names_colors,
            self.current_layer_object_categories,
            self.current_layer_categories
        ) = self.data.prepare_array(coloring)
        if self.unselected_objs is None:
            ignore = []
        else:
            ignore = self.unselected_objs
        width = self.bbox_thickness.get() if width is None else width
        alpha = self.mask_alpha.get() if alpha is None else alpha
        label_size = self.label_size.get() if label_size is None else label_size
        # Compose image
        self.compose_image(
            image_current_layer_array=image_current_layer_array,
            regions=regions,
            names_colors=names_colors,
            bboxes_on=bboxes_on,
            labels_on=labels_on,
            masks_on=masks_on,
            ignore=ignore,
            width=width,
            alpha=alpha,
            label_size=label_size,
        )
        # Prepare PIL image for Tkinter
        if (cwidth := self.image_panel._canvas.winfo_width()) > 512 and (cheight := self.image_panel._canvas.winfo_height()) > 512:
            img = self.current_composed_image.resize(
                (min(cwidth, cheight), min(cwidth, cheight)), resample=Image.Resampling.LANCZOS)
        else:
            img = self.current_composed_image.resize(
                (512, 512), resample=Image.Resampling.LANCZOS)
        w, h = img.size
        img = ImageTk.PhotoImage(img)
        # Set image as current
        self.image_panel.create_image(0, 0, image=img)
        self.image_panel.image = img
        self.image_panel.reset(canvwidth=w, canvheight=h)
        # Update statusbar vars
        self.file_count_status.set(
            f"{self.data.images.n + 1}/{self.data.images.max}")
        self.file_name_status.set(
            f"filename: {os.path.basename(self.data.current_image)}")
        self.layer_status.set(
            f"{self.data.current_layer + 1}/{self.data.max_layer + 1}")
        self.nobjects_status.set(
            f"objects: {len(self.current_layer_object_categories)}")
        self.ncategories_status.set(
            f"categories: {len(self.current_layer_categories)}")
        # Update Objects panel
        self.update_category_box()
        self.update_object_box()

    def exit(self, event=None):
        print_info("Exiting...")
        self.root.quit()

    def next_img(self, event=None):
        self.data.next_image()
        self.set_locals()
        self.selected_cats_ids = None
        self.selected_objs_ids = None
        self.unselected_objs = None
        self.update_img(local=False)

    def prev_img(self, event=None):
        self.data.previous_image()
        self.set_locals()
        self.selected_cats_ids = None
        self.selected_objs_ids = None
        self.unselected_objs = None
        self.update_img(local=False)

    def next_layer(self, event=None):
        self.set_locals()
        self.selected_cats_ids = None
        self.selected_objs_ids = None
        self.unselected_objs = None
        if self.data.current_layer < self.data.max_layer:
            self.data.current_layer += 1
            self.update_img(local=False)
        else:
            self.data.current_layer = 0
            self.next_img()

    def prev_layer(self, event=None):
        self.set_locals()
        self.selected_cats_ids = None
        self.selected_objs_ids = None
        self.unselected_objs = None
        if self.data.current_layer > 0:
            self.data.current_layer -= 1
            self.update_img(local=False)
        else:
            self.data.current_layer = 0
            self.prev_img()

    def save_image(self, event=None):
        """Saves composed image as png file."""
        # Initial (original) file name
        initialfile = os.path.basename(self.data.current_image)
        # TODO: Add more formats, at least jpg (RGBA -> RGB)?
        filetypes = (("png files", "*.png"), ("all files", "*.*"))
        # By default save as png file
        defaultextension = ".png"
        file = filedialog.asksaveasfilename(
            initialfile=initialfile,
            filetypes=filetypes,
            defaultextension=defaultextension,
        )
        # If not canceled:
        if file:
            self.current_composed_image.save(file)

    def menu_view_bboxes(self):
        self.bboxes_on_local = self.bboxes_on_global.get()
        self.bbox_slider_status_update()
        self.update_img()

    def menu_view_labels(self):
        self.labels_on_local = self.labels_on_global.get()
        self.label_slider_status_update()
        self.update_img()

    def menu_view_masks(self):
        self.masks_on_local = self.masks_on_global.get()
        self.masks_slider_status_update()
        self.update_img()

    def menu_view_coloring(self):
        self.coloring_on_local = self.coloring_on_global.get()
        self.update_img()

    def toggle_bboxes(self, event=None):
        self.bboxes_on_local = not self.bboxes_on_local
        self.bbox_slider_status_update()
        self.update_img()

    def toggle_labels(self, event=None):
        self.labels_on_local = not self.labels_on_local
        self.label_slider_status_update()
        self.update_img()

    def toggle_masks(self, event=None):
        self.masks_on_local = not self.masks_on_local
        self.update_img()

    def toggle_all(self, event=None):
        # Toggle only when focused on image
        if event.widget.focus_get() is self.objects_panel.category_box:
            return
        if event.widget.focus_get() is self.objects_panel.object_box:
            return
        # What to toggle
        var_list = [self.bboxes_on_local,
                    self.labels_on_local, self.masks_on_local]
        # if any is on, turn them off
        if True in set(var_list):
            self.bboxes_on_local = False
            self.labels_on_local = False
            self.masks_on_local = False
        # if all is off, turn them on
        else:
            self.bboxes_on_local = True
            self.labels_on_local = True
            self.masks_on_local = True
        # Update sliders
        self.update_sliders_state()
        # Update image with updated vars
        self.update_img()

    def update_category_box(self):
        ids = self.current_layer_categories
        names = [self.data.categories[i][0] for i in ids]
        self.category_box_content.set(
            [" ".join([str(i), str(n)]) for i, n in zip(ids, names)])
        self.objects_panel.category_box.selection_clear(0, tk.END)
        if self.selected_cats_ids is not None:
            for i in self.selected_cats_ids:
                self.objects_panel.category_box.select_set(i)
        else:
            self.objects_panel.category_box.select_set(0, tk.END)

    def select_category(self, event):
        # Get selected_cats_ids from user
        self.selected_cats_ids = self.objects_panel.category_box.curselection()
        self.selected_cats = [self.current_layer_categories[i]
                              for i in self.selected_cats_ids]
        # Set unselected_objs
        self.unselected_objs = []
        for oc in self.current_layer_object_categories:
            if oc in self.selected_cats:
                self.unselected_objs.append(0)
            else:
                self.unselected_objs.append(1)
        self.selected_objs_ids = None
        self.update_img()

    def update_object_box(self):
        ids = self.current_layer_object_categories
        names = [self.data.categories[i][0] for i in ids]
        self.object_box_content.set(
            [" ".join([str(i), str(n)]) for i, n in enumerate(names)])
        self.objects_panel.object_box.selection_clear(0, tk.END)
        if self.selected_objs_ids is not None:
            for i in self.selected_objs_ids:
                self.objects_panel.object_box.select_set(i)
        else:
            self.objects_panel.object_box.select_set(0, tk.END)

    def select_object(self, event):
        # Get selected_objs_ids from user
        self.selected_objs_ids = self.objects_panel.object_box.curselection()
        self.unselected_objs = []
        for i in range(len(self.current_layer_object_categories)):
            if i in self.selected_objs_ids:
                self.unselected_objs.append(0)
            else:
                self.unselected_objs.append(1)
        self.selected_cats_ids = None
        self.update_img()

    def update_sliders_state(self):
        self.bbox_slider_status_update()
        self.label_slider_status_update()
        self.masks_slider_status_update()

    def bbox_slider_status_update(self):
        self.sliders.bbox_slider.configure(
            state=tk.NORMAL if self.bboxes_on_local else tk.DISABLED)

    def label_slider_status_update(self):
        self.sliders.label_slider.configure(
            state=tk.NORMAL if self.labels_on_local else tk.DISABLED)

    def masks_slider_status_update(self):
        self.sliders.mask_slider.configure(
            state=tk.NORMAL if self.masks_on_local else tk.DISABLED)

    def bind_events(self):
        """Binds events."""
        # Navigation
        self.root.bind("<Left>", self.prev_img)
        self.root.bind("<a>", self.prev_img)
        self.root.bind("<Right>", self.next_img)
        self.root.bind("<d>", self.next_img)
        self.root.bind("<Up>", self.prev_layer)
        self.root.bind("<w>", self.prev_layer)
        self.root.bind("<Down>", self.next_layer)
        self.root.bind("<s>", self.next_layer)
        self.root.bind("<Control-q>", self.exit)
        self.root.bind("<Control-w>", self.exit)
        # Files
        self.root.bind("<Control-s>", self.save_image)
        # View Toggles
        self.root.bind("<b>", self.toggle_bboxes)
        self.root.bind("<Control-b>", self.toggle_bboxes)
        self.root.bind("<l>", self.toggle_labels)
        self.root.bind("<Control-l>", self.toggle_labels)
        self.root.bind("<m>", self.toggle_masks)
        self.root.bind("<Control-m>", self.toggle_masks)
        self.root.bind("<space>", self.toggle_all)
        # Objects Panel
        self.objects_panel.category_box.bind(
            "<<ListboxSelect>>", self.select_category)
        self.objects_panel.object_box.bind(
            "<<ListboxSelect>>", self.select_object)
        self.image_panel.bind(
            "<Button-1>", lambda e: self.image_panel.focus_set())


def print_info(message: str):
    logging.info(message)


def main():
    print_info("Starting...")
    args = parser.parse_args()
    root = tk.Tk()
    root.title("NIFTI Viewer")
    if not args.images or not args.annotations:
        root.geometry("300x150")  # app size when no data is provided
        messagebox.showwarning(
            "Warning!", "Nothing to show.\nPlease specify a path to the COCO dataset!")
        print_info("Exiting...")
        root.destroy()
        return
    data = Data(args.images, args.annotations, args.categories)
    statusbar = StatusBar(root)
    sliders = SlidersBar(root)
    objects_panel = ObjectsPanel(root)
    menu = Menu(root)
    image_panel = ImagePanel(root)
    Controller(data, root, image_panel, statusbar,
               menu, objects_panel, sliders)
    root.mainloop()


if __name__ == "__main__":
    main()
