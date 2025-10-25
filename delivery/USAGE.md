# Packaging the timeframe update yourself

Binary archives are not stored in the repository, but you can generate the same bundle locally. The helper script below produces a zip file that mirrors the current working tree so you can copy the patched files into another checkout without handling merge conflicts.

1. Run the packaging script (it lives alongside this guide):

   ```bash
   python delivery/create_update_archive.py
   ```

   This creates `delivery/latest_strategy_update.zip` in place. The archive only includes files tracked by git, so temporary logs and CSV exports are ignored.

   If your destination rejects binary uploads, emit a base64-encoded archive instead:

   ```bash
   python delivery/create_update_archive.py --format base64
   ```

   The resulting `delivery/latest_strategy_update.zip.b64` is plain text and can be pasted into systems that disallow raw binary attachments.

2. Extract the archive on the target machine and overwrite the existing files.

   * **Windows PowerShell**

     ```powershell
     Expand-Archive -Path .\delivery\latest_strategy_update.zip -DestinationPath .\delivery\latest_strategy_update -Force
     ```

   * **macOS / Linux**

     ```bash
     unzip delivery/latest_strategy_update.zip -d delivery/latest_strategy_update
     ```

   Copy the extracted files into your project directory (preserve the directory structure so Python modules, tests, and workflow files land in the right place).

   When working with the text archive, use the Python helper to decode and unpack everything without needing external tools:

   ```bash
   python delivery/extract_update_archive.py delivery/latest_strategy_update.zip.b64 --dest delivery/latest_strategy_update
   ```

   The helper also works with the binary `.zip` file, so it is safe to use in any environment that has Python 3 available.
