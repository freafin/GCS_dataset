import os

def write_filenames_to_txt(folder_path, output_txt_path):
    """
    Writes all filenames (without extensions) in the specified folder to a text file.
    Each line in the output file will contain one filename.
    """
    try:
        filenames = os.listdir(folder_path)
        with open(output_txt_path, "w") as f:
            for name in filenames:
                base_name, _ = os.path.splitext(name)
                f.write(base_name + "\n")
        print(f"Saved filenames to: {output_txt_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Replace with your actual folder path
    folder_path = "insert_path"
    output_txt_path = "insert_path"
    write_filenames_to_txt(folder_path, output_txt_path)