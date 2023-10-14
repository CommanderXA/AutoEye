import os
import sys
import pandas as pd

args = sys.argv

classnames = {
    "0-correct": "0",
    "fictivniye(fictitious)": "1",
    "1-not-on-the-brake-stand": "1",
    "2-from-the-screen": "2",
    "3-from-the-screen+photoshop": "3",
    "4-photoshop": "4",
}

if not len(args) == 2:
    print("format: 'python create_csv.py train' ot 'python create_csv.py test'")
    sys.exit()


def read_files(directory):
    if args[1] == "test":
        df = pd.DataFrame(columns=["file_index"])
        for root, dirs, files in os.walk(directory):
            for file in files:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [{"file_index": os.path.join(root, file)}], index=None
                        ),
                    ],
                    ignore_index=False,
                )
        return df

    labels = ["path", "class", "subclass"]
    df = pd.DataFrame(columns=labels)

    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            classname = classnames[os.path.basename(root)]
            subclass = "0"

            if (
                not classname == "0"
            ):  # if class is not '0', then change class to parent directory name, and subclass to this dirs name
                subclass = classname
                classname = os.path.dirname(root)
                classname = classnames[os.path.basename(classname)]

            new_row = {"path": path, "class": classname, "subclass": subclass}
            df = pd.concat([df, pd.DataFrame([new_row], index=None)], ignore_index=True)

    return df


df = None

if args[1] == "train":
    df = read_files("./data/case3-datasaur-photo/techosmotr/techosmotr/train/")

if args[1] == "test":
    df = read_files("./data/case3-datasaur-photo/techosmotr/techosmotr/test/")


df = df.reset_index()
print(df)

df.to_csv("./data/data.csv", sep=";", index=False)
