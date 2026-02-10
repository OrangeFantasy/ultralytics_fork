import torch

if __name__ == "__main__":
    replaces = {
        "model.21.cv4.0.0.": "model.21.cv4.0.",
        "model.21.cv4.0.1.": "model.21.cv4.1.",
        "model.21.cv4.0.2.": "model.21.cv4_kpts.0.",
        "model.21.cv4.1.0.": "model.21.cv4.1.",
        "model.21.cv4.1.1.": "model.21.cv4.2.",
        "model.21.cv4.1.2.": "model.21.cv4_kpts.1.",
        "model.21.cv4.2.0.": "model.21.cv4.2.",
        "model.21.cv4.2.1.": "model.21.cv4.3.",
        "model.21.cv4.2.2.": "model.21.cv4_kpts.2.",
    }
    replaces = {
        'cv4.0.0.0.': 'cv4.0.0.', 
        'cv4.0.0.1.': 'cv4.0.1.',
        'cv4.0.1.0.': 'cv4.1.0.',
        'cv4.0.1.1.': 'cv4.1.1.',
        'cv4.0.2.0.': 'cv4.2.0.',
        'cv4.0.2.1.': 'cv4.2.1.',
        'cv4.0.0.2.': 'cv4_kpts.0.',
        'cv4.0.1.2.': 'cv4_kpts.1.',
        'cv4.0.2.2.': 'cv4_kpts.2.',
    }

    # checkpoint = "/data4/yuanchengzhi/projects/yolo/MultiTaskDetector/runs/sports/448x768_2H_solid_260111_300/weights/best.fp16.state_dict.pt"
    checkpoint = "/data4/yuanchengzhi/projects/yolo/MultiTaskDetector/runs/mdetectors_qrcode/qrcode_0112/weights/best.fp16.state_dict.pt"
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)

    new_checkpoint = { 
        "epoch": checkpoint["epoch"],
        "date": checkpoint["date"]
    }

    new_state_dict = { }
    for key, value in checkpoint['state_dict'].items():
        for old_str, new_str in replaces.items():
            if old_str in key:
                new_key = key.replace(old_str, new_str)
                print(f"{key} -> {new_key}")
                key = new_key
        new_state_dict[key] = value
    new_checkpoint["state_dict"] = new_state_dict
    torch.save(new_checkpoint, "./converted.pt")
