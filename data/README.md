# Data Layout

The image datasets are not committed to Git because they are large and contain locally collected media.

Expected folders:

```text
data/
  raw/
    clean_dataset/
      Cocacola_classic/
      Sprite/
      Redbull_Classic/
      ...
  processed/
    clean_dataset_split/
      train/
      val/
      test/
  robot_captured/
    Cocacola_classic/
    Sprite/
    Redbull_Classic/
  robot_split/
    train/
    val/
    test/
```

Each class folder should contain image files such as `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`.

