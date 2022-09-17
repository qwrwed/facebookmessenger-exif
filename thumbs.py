import argparse
import json
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
    show_images: bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backup", action="store_true")
    parser.add_argument("-i", "--input-dir", default=os.getcwd(), type=_normpath)
    parser.add_argument("-s", "--show-images", action="store_true")
    return parser.parse_args(namespace=ProgramArgsNamespace)


def getAspectRatio(image: "np.ndarray"):
    h, w, _ = image.shape
    return w / h


def hasSameAspectRatio(image1: "np.ndarray", image2: "np.ndarray"):
    return abs(getAspectRatio(image1) - getAspectRatio(image2)) < DIFFERENCE_THRESHOLD


if __name__ == "__main__":
    args = get_args()
    exiftool_set_params = ["-P"]
    if not args.backup:
        exiftool_set_params.append("-overwrite_original")

    thumbnail_paths = list(args.input_dir.glob("**/thumbnails/*.jpg"))
    already_mapped_videos = set()
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

                if args.show_images:
                    thumbnail_window_title = f"reference | thumb {thumbnail_path}"
                    video_window_title = "video"
                    cv2.namedWindow(thumbnail_window_title)
                    cv2.namedWindow(video_window_title)
                    cv2.moveWindow(thumbnail_window_title, 0, 0)
                    cv2.moveWindow(video_window_title, thumbnail_image.shape[1] + 10, 0)
                    cv2.imshow(thumbnail_window_title, thumbnail_image)
                    cv2.imshow(video_window_title, firstframe_image)

                if difference < DIFFERENCE_THRESHOLD:
                    tqdm.write(
                        f"{Path(*thumbnail_path.parts[-4:])} => {Path(*video_path.parts[-2:])}"
                    )
                    already_mapped_videos.add(video_path)
                    mapped = True
                    if args.show_images:
                        cv2.imshow(video_window_title, firstframe_image)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()
                    break
                else:
                    if args.show_images:
                        cv2.waitKey(1)

            if mapped:
                thumbnail_metadata = et.get_metadata(thumbnail_path)[0]
                video_metadata = et.get_metadata(video_path)[0]
                video_date_metadata = {
                    tag: value
                    for tag, value in video_metadata.items()
                    if tag in thumbnail_metadata and tag.endswith("Date")
                }
                et.set_tags(thumbnail_path, tags=video_date_metadata)
                thumbnail_metadata_new = et.get_metadata(thumbnail_path)[0]
            else:
                print(f"{Path(*thumbnail_path.parts[-4:])} => {None}")

