"""
STEP 2 — Upload to Google Drive (manual)
=========================================
After running step1_preprocess.py you'll have brain_tumor_dataset.zip
in the project root (~100-150 MB).  Upload it manually using any of the
methods below, then open the Colab notebook.

──────────────────────────────────────────────────────────────────────
Method A — Google Drive web UI (simplest)
──────────────────────────────────────────────────────────────────────
1. Open https://drive.google.com
2. Create a folder called  BrainTumor
3. Drag and drop  brain_tumor_dataset.zip  into it

──────────────────────────────────────────────────────────────────────
Method B — Google Drive desktop app (macOS)
──────────────────────────────────────────────────────────────────────
If you have Google Drive for Desktop installed:
1. Open Finder → Google Drive (mirrored folder)
2. Create a  BrainTumor  folder
3. Copy  brain_tumor_dataset.zip  into it — it syncs automatically

──────────────────────────────────────────────────────────────────────
Method C — rclone CLI (fastest for large files)
──────────────────────────────────────────────────────────────────────
Install:
    brew install rclone

One-time config (creates a 'gdrive' remote):
    rclone config
    # choose: n (new remote) → name: gdrive → type: drive → follow OAuth prompts

Upload:
    rclone copy brain_tumor_dataset.zip gdrive:BrainTumor/ --progress

Verify:
    rclone ls gdrive:BrainTumor/

──────────────────────────────────────────────────────────────────────
After uploading
──────────────────────────────────────────────────────────────────────
In the Colab notebook (Cell 3) set:
    ZIP_ON_DRIVE = '/content/drive/MyDrive/BrainTumor/brain_tumor_dataset.zip'

Then run all cells.
"""

print(__doc__)
