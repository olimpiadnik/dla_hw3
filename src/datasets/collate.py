
import torch
from torch.utils.data._utils.collate import default_collate


def collate_fn(dataset_items):
    if len(dataset_items) == 0:
        return {}

    first = dataset_items[0]

    if "data_object" in first:
        batch = {}
        batch["data_object"] = torch.stack(
            [item["data_object"] for item in dataset_items], dim=0
        )

        if "labels" in first:
            labels = [item["labels"] for item in dataset_items]
            if isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels, dim=0)
            batch["labels"] = labels

        extra_keys = set(first.keys()) - {"data_object", "labels"}
        for key in extra_keys:
            values = [item[key] for item in dataset_items]
            try:
                batch[key] = default_collate(values)
            except Exception:
                batch[key] = values

        return batch

    if "audio" in first and "mel" in first:
        batch = {}

        batch["audio"] = torch.stack(
            [item["audio"] for item in dataset_items], dim=0
        )

        batch["mel"] = torch.stack(
            [item["mel"] for item in dataset_items], dim=0
        )

        if "path" in first:
            batch["path"] = [item["path"] for item in dataset_items]

        if "text" in first:
            batch["text"] = [item["text"] for item in dataset_items]

        return batch

    raise KeyError(
        f"Unsupported batch format in collate_fn. Keys: {list(first.keys())}"
    )
