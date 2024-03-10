

def read_all_text(data_path: str) -> list:
    with open(data_path, "r") as fo:
        data = fo.readlines()

    return [sent.strip() for sent in data]
