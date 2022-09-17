import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import exiftool
from tqdm import tqdm

DIFFERENCE_THRESHOLD = 0.01


def _normpath(path):
    return Path(os.path.normpath(path))


class ProgramArgsNamespace(argparse.Namespace):
    backup: bool
    input_dir: Path
    show_images_unsafe: bool


def get_args():
    parser = argparse.ArgumentParser(
        description="""
            Apply metadata to video thumbnails from Facebook messenger data.
            Assumes any .jpg image in any 'thumbnails' folder is a thumbnail,
            and has a corresponding .mp4 video in its parent folder.
        """
    )
    parser.add_argument(
        "input_dir",
        metavar="FOLDER",
        type=_normpath,
        help="Folder to search (recursively) for thumbnails in",
    )
    parser.add_argument(
        "-s",
        "--show-images-unsafe",
        action="store_true",
        help="Display image matching process (may cause entire OS to hang)",
    )
    parser.add_argument(
        "-b",
        "--backup",
        action="store_true",
        help="Backup modified files",
    )
    return parser.parse_args(namespace=ProgramArgsNamespace)


def getAspectRatio(image: "np.ndarray"):
    h, w, _ = image.shape
    return w / h


def hasSameAspectRatio(image1: "np.ndarray", image2: "np.ndarray"):
    return abs(getAspectRatio(image1) - getAspectRatio(image2)) < DIFFERENCE_THRESHOLD


if __name__ == "__main__":
    args = get_args()
    input_dir = args.input_dir
    pattern = "*.jpg" if input_dir.parts[-1] == "thumbnails" else "**/thumbnails/*.jpg"
    thumbnail_paths = list(args.input_dir.glob(pattern))

    already_mapped_videos = set()

    exiftool_params_write = []
    if not args.backup:
        exiftool_params_write.append("-overwrite_original")

    with exiftool.ExifToolHelper() as et:
        for thumbnail_path in tqdm(thumbnail_paths):

            mapped = False
            thumbnail_image: "np.ndarray" = cv2.imread(str(thumbnail_path))

            thumbnail_dir = thumbnail_path.parent
            video_dir = thumbnail_path.parent.parent

            video_paths = video_dir.glob("*.mp4")
            for video_path in video_paths:
                if video_path in already_mapped_videos:
                    continue

                vidcap = cv2.VideoCapture(str(video_path))
                success: bool
                firstframe_image: "np.ndarray"
                success, firstframe_image = vidcap.read()

                if not hasSameAspectRatio(thumbnail_image, firstframe_image):
                    continue
                firstframe_image = cv2.resize(
                    firstframe_image, thumbnail_image.shape[:2][::-1]
                )

                difference = (
                    1
                    - cv2.matchTemplate(
                        thumbnail_image, firstframe_image, cv2.TM_CCOEFF_NORMED
                    )[0][0]
                )

                if args.show_images_unsafe:
                    thumbnail_window_title = f"reference | thumb {thumbnail_path}"
                    video_window_title = "video"
                    cv2.namedWindow(thumbnail_window_title)
                    cv2.namedWindow(video_window_title)
                    cv2.moveWindow(thumbnail_window_title, 0, 0)
                    cv2.moveWindow(video_window_title, thumbnail_image.shape[1] + 10, 0)
                    cv2.imshow(thumbnail_window_title, thumbnail_image)
                    cv2.imshow(video_window_title, firstframe_image)

                if difference < DIFFERENCE_THRESHOLD:
                    # tqdm.write(
                    #     f"{Path(*thumbnail_path.parts[-4:])} => {Path(*video_path.parts[-2:])}"
                    # )
                    already_mapped_videos.add(video_path)
                    mapped = True
                    if args.show_images_unsafe:
                        cv2.imshow(video_window_title, firstframe_image)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()
                    break
                else:
                    if args.show_images_unsafe:
                        cv2.waitKey(1)

            if mapped:
                thumbnail_metadata = et.get_metadata(thumbnail_path)[0]
                video_metadata = et.get_metadata(video_path)[0]
                video_date_metadata = {
                    tag: value
                    for tag, value in video_metadata.items()
                    if tag in thumbnail_metadata and tag.endswith("Date")
                }
                date = video_date_metadata["File:FileCreateDate"]
                video_date_metadata.update(
                    {
                        "EXIF:DateTimeOriginal": date,
                        "XMP:CreationDate": date,
                    }
                )
                et.set_tags(
                    thumbnail_path,
                    tags=video_date_metadata,
                    params=exiftool_params_write,
                )
                thumbnail_metadata_new = et.get_metadata(thumbnail_path)[0]
            else:
                # tqdm.write(f"{Path(*thumbnail_path.parts[-4:])} <= {None}")
                pass
