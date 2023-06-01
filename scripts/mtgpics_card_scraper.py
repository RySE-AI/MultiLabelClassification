import requests
import logging

from bs4 import BeautifulSoup
from argparse import ArgumentParser
from urllib.parse import urlparse
from tqdm import tqdm
from pathlib import Path

from typing import Dict

logger = logging.getLogger(__name__)


# TODO: Add Logger
# TODO: Add docstrings
# TODO: Multiprocessing?
def scrape_card_urls(soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
    img_urls = {}
    for img_item in soup.find_all("img"):
        img_item: dict = img_item.attrs

        if "src" not in img_item:
            continue
        img_url = img_item["src"]

        # Most of the single card images of a deck startswith "../"
        if img_url.startswith("../"):
            splitted_url = img_url.split("/")
            deck_name = splitted_url[-2]
            full_card_name = splitted_url[-1].split(".")[0]

            if "alt" in img_item:
                card_name = img_item["alt"].split(" -")[0].replace(" ", "_")
            else:
                card_name = "na"  # No Alt

            full_card_name = f"{deck_name}_{full_card_name}_{card_name}"
            img_url = base_url + img_url.replace("../", "")
            img_urls[full_card_name] = img_url

    return img_urls


def save_card_images(img_urls: Dict, save_dir: Path) -> None:
    if not save_dir.is_dir():
        print(f"Creating new directory at {save_dir.resolve()}")
        save_dir.mkdir(parents=True)

    for full_card_name, img_url in tqdm(img_urls.items()):
        img_response = requests.get(img_url.replace("reg", "big"))
        file_name = f"{full_card_name}.jpg"
        with open(save_dir / file_name, "wb") as f:
            f.write(img_response.content)


def get_data(url: str) -> str:
    r = requests.get(url)
    return r.text


def main(args) -> None:
    """Use this script to scrape the images of a magic card deck.

    For example to get the m21 deck:
    python scripts/mtgpics_card_scraper.py -du "https://www.mtgpics.com/set?set=318" --save-dir "./data/images/m21"

    The images will be named after there alt content. The deckname
    is used as a prefix, followed by the card id.
    Content, both literal and graphical, is copyrighted by Wizards of the Coast.
    This is a fan project with non-comercial usage!
    """
    url_parser = urlparse(args.deck_url)
    base_url = f"{url_parser.scheme}://{url_parser.netloc}/"

    request_text = get_data(args.deck_url)
    soup = BeautifulSoup(request_text, "html.parser")
    img_urls = scrape_card_urls(soup=soup, base_url=base_url)
    save_card_images(img_urls=img_urls, save_dir=Path(args.save_dir))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-du",
        "--deck-url",
        help="URL to the deck you want to download",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--save-dir", help="Path to save images", default="./", type=str
    )
    args = parser.parse_args()

    main(args)
