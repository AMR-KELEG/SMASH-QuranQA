# This Python file uses the following encoding: utf-8

import os
import re
from bs4 import BeautifulSoup
from data.wikipedia_raw_files.bible_quran_persons import bible_quran_persons_table
from data.wikipedia_raw_files.quran_persons import lists_of_quran_persons
from data.wikipedia_raw_files.quran_animals import list_of_animals


def get_augmented_value(person):
    """Generate different case inflections for some words"""
    if re.search(r"^ذو\s", person):
        return re.sub("^ذو", "ذي", person)
    if re.search(r"^أبو\s", person):
        return re.sub("^أبو", "أبي", person)
    if re.search(r"^ال.*ون$", person):
        return re.sub("ون$", "ين", person)
    return None


if __name__ == "__main__":
    soup = BeautifulSoup(bible_quran_persons_table, "html.parser")

    # Parse second column in each row skipping the header
    persons = [
        person.find_all("td")[1].get_text().strip()
        for person in soup.find_all("tr")[1:]
    ]

    # Remove text between ()
    persons = [re.sub(r"[(].*[)]", "", p).strip() for p in persons]

    # Split multiple variants of the same name
    persons = sum(
        [
            [p]
            if not any([mark in p for mark in ["أو", "،"]])
            else p.split("أو")
            if "أو" in p
            else p.split("،")
            for p in persons
        ],
        [],
    )
    # Remove extra whitespaces
    persons = [p.strip() for p in persons]
    persons = sorted(set(persons))

    for list_of_persons in lists_of_quran_persons:
        soup = BeautifulSoup(list_of_persons, "html.parser")

        for li in soup.find_all("li"):
            persons.append(li.find_all("a")[0].text)

    persons = (
        persons
        + [get_augmented_value(p) for p in persons if get_augmented_value(p)]
        + ["عيسى ابن مريم", "قوم تبع"]
        + ["ثمود", "إرم"]
        + ["الأسباط"]
        + ["يهود", "نصاري"]
        + ["ملائكة", "شياطين"]
    )
    # Remove "تبع" as it overlaps with the verb "تبع" causing False positives
    persons = [p for p in persons if p != "تبع"]
    persons = sorted(set([p.strip() for p in persons]))

    soup = BeautifulSoup(list_of_animals, "html.parser")

    animals = []
    for li in soup.find_all("li"):
        animal_name = li.text.split(" - ")[0].split(":")[-1].strip()
        # Drop "ال" from the beginning of the animal names
        if animal_name.startswith("ال"):
            animal_name = animal_name[2:]
        animals.append(animal_name)

    # Export the lists to txt files
    os.makedirs("data", exist_ok=True)
    with open("data/persons.txt", "w") as f:
        f.write("\n".join(persons))

    with open("data/animals.txt", "w") as f:
        f.write("\n".join(animals))
