from enum import Enum
from PIL import Image, ImageTk
from typing import Dict

from logic.sokoban_map import MapObject, Ground


MAP_OBJECT_IMAGE_PATHS : Dict[MapObject, str] = {
    MapObject.CRATE: "assets/crate.png",
    MapObject.SOKOBAN: "assets/sokoban.png",
}

GROUND_IMAGE_PATHS : Dict[Ground, str] = {
    Ground.FLOOR: "assets/floor.png",
    Ground.WALL: "assets/wall.png",
    Ground.STORAGE: "assets/storage.png",
}

CLEAR_IMAGE_PATH : str = "assets/clear.png"
EDIT_IMAGE_PATH : str = "assets/edit.png"

def load_ground_images() -> Dict[Ground, Image.Image]:
    images : Dict[Ground, Image.Image] = {}
    for ground, path in GROUND_IMAGE_PATHS.items():
        images[ground] = Image.open(path)
    return images

def load_map_object_images() -> Dict[MapObject, Image.Image]:
    images : Dict[MapObject, Image.Image] = {}
    for obj, path in MAP_OBJECT_IMAGE_PATHS.items():
        images[obj] = Image.open(path)
    return images

def load_clear_image() -> Image.Image:
    return Image.open(CLEAR_IMAGE_PATH)

def load_edit_image() -> Image.Image:
    return Image.open(EDIT_IMAGE_PATH)

def resize_image(image: Image.Image, size: int) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(image.resize((size, size), Image.LANCZOS))