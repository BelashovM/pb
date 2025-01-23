import xml.etree.ElementTree as ET
import os

def extract_and_save_data(file_path, tags):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for i in tags:
        for k in root.findall(f".//{i}"):
            if k.text:
                data.append((i, k.text.strip()))

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    dataf = os.path.join(desktop_path, "data.txt")

    with open(dataf, 'w', encoding='utf-8') as file:
        for tag, value in data:
            file.write(f"{tag}\t{value}\n")

    print(f"Данные сохранены в файл: {dataf}")


# Основной код
if __name__ == "__main__":
    xml_file = "E:\Settings.xml"

    tags_to_search = ['Width','Height','StartPoint','EndPoint','Step','Accuracy','AccuracyAttempts','AdjustmentPoint','Delay',
                      'Averages','Sensetivity','TimeConstant','ConfMemLength','SplitterPosition']  # Замените на ваши теги

    extract_and_save_data(xml_file, tags_to_search)
