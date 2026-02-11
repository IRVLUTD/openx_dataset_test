# import json
# file_name = '/metadisk/label-studio/project8.json'
# with open(file_name) as f:
#     annotation = json.load(f)
#     print(annotation)


# results = []
# for anno in annotation:
#     result = dict()
#     result['id'] = anno['id']
#     result['img'] = anno['data']['image']
import json
import argparse
import os
import numpy as np
from label_studio_format import rle_to_mask, mask_to_polygon, visualize_conversion


def load_annotation_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def convert_to_coco_format(data):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_ids = {}
    num_classes = 0
    annotation_id = 0

    for item in data:
        image_id = item['id']
        annotations = item['annotations']
        image_path = item['data']['image']
        image_folder = image_path.split('/')[-2][3:]
        image_name = image_path.split('/')[-1]
        image_name = os.path.join(image_folder, image_name)

        # Process image information
        width = annotations[0]['result'][0]['original_width']
        height = annotations[0]['result'][0]['original_height']
        coco_output["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"{image_name}"  # Adjust filename if necessary
        })

        for ann in annotations:
            for result in ann['result']:
                if result['type'] != 'brushlabels':
                    continue
                result_id = result['id']
                label = result['value']['brushlabels'][0]
                print(label)
                referring_text = next(
                    (res['value']['text'][0] for res in ann['result'] if
                     res['id'] == result_id and 'text' in res['value']),
                    ""
                )

                # Add category if it does not exist
                if label not in category_ids:
                    category_id = num_classes
                    num_classes += 1
                    category_ids[label] = category_id
                    coco_output["categories"].append({
                        "id": category_id,
                        "name": label,
                        "supercategory": "none"
                    })
                else:
                    category_id = category_ids[label]

                # Decode RLE if available
                rle_counts = result['value']['rle']
                # result and value are from the label studio JSON format
                binary_mask = rle_to_mask(rle_counts, height, width)
                # coco uses polygons for segmentation
                polygons = mask_to_polygon(binary_mask)
                #visualize_conversion(binary_mask, polygons)
                #print(np.unique(binary_mask))
                #mask_values = np.unique(binary_mask)

                # Convert RLE mask to bounding box
                # Calculate bounding box from binary mask
                mask_pixels = np.where(binary_mask == 255)
                if mask_pixels[0].size > 0:
                    x_min, y_min = np.min(mask_pixels[1]), np.min(mask_pixels[0])
                    x_max, y_max = np.max(mask_pixels[1]), np.max(mask_pixels[0])
                    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": polygons,  # Polygon segmentation format
                        "area": int(np.sum(binary_mask)),
                        "iscrowd": 0,
                        "result_id": result_id,
                        "referring": referring_text
                    })
                    annotation_id += 1

    return coco_output


def save_coco_format(coco_output, output_path):
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=4)


def main(input_path: str, output_path: str):
    data = load_annotation_data(input_path)
    coco_format_data = convert_to_coco_format(data)
    save_coco_format(coco_format_data, output_path)
    print(f"COCO format annotations saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Label Studio annotations to COCO format"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input Label Studio annotation JSON file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save COCO-formatted annotation JSON file",
    )

    # example
    #input_path = '/metadisk/label-studio/raw_annotation_label_studio/project_01.json'  # Path to your input annotation file
    #output_path = '/metadisk/label-studio/referring_coco_annotation/project_01_coco.json'  # Path to save COCO-formatted annotations

    args = parser.parse_args()
    main(args.input_path, args.output_path)