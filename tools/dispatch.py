import sys
import subprocess
import os
import shutil


def main():
    # --- Configuration ---
    # Determine paths relative to this script (Linux_Dist/tools/dispatch.py)
    script_path = os.path.abspath(__file__)
    tools_dir = os.path.dirname(script_path)
    root_dir = os.path.dirname(tools_dir)

    # Path to av1an script
    av1an_script = os.path.join(root_dir, "Auto-Boost-Av1an.py")

    # Locate MediaInfo (Linux System Path)
    mediainfo_exe = shutil.which("mediainfo")

    # --- Argument Parsing ---
    # Find input file (-i or --input)
    args = sys.argv[1:]
    input_file = None

    for idx, arg in enumerate(args):
        if arg in ("-i", "--input") and idx + 1 < len(args):
            input_file = args[idx + 1]
            break

    # --- BT.709 Detection via MediaInfo ---
    is_bt709 = False

    # Flags to track if we found the specific BT.709 values
    found_primaries = False
    found_transfer = False
    found_matrix = False

    if input_file and os.path.exists(input_file):
        if mediainfo_exe:
            try:
                # Run MediaInfo on the file
                cmd = [mediainfo_exe, input_file]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )

                if result.returncode == 0:
                    # Parse the text output line by line
                    for line in result.stdout.splitlines():
                        if ":" not in line:
                            continue

                        # Split into Key : Value
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()

                        # Check strict equality for the values
                        if key == "Color primaries" and value == "BT.709":
                            found_primaries = True
                        elif key == "Transfer characteristics" and value == "BT.709":
                            found_transfer = True
                        elif key == "Matrix coefficients" and value == "BT.709":
                            found_matrix = True

                    # All three must be present and equal to BT.709
                    if found_primaries and found_transfer and found_matrix:
                        is_bt709 = True
                        print("[Dispatch] MediaInfo confirmed full BT.709 source.")
                    else:
                        print(
                            f"[Dispatch] MediaInfo results - Primaries: {found_primaries}, Transfer: {found_transfer}, Matrix: {found_matrix}. Not pure BT.709."
                        )
                else:
                    print("[Dispatch] Warning: MediaInfo returned an error.")
            except Exception as e:
                print(f"[Dispatch] Warning: MediaInfo execution failed: {e}")
        else:
            print(f"[Dispatch] Warning: mediainfo not found in PATH.")
            print("[Dispatch] Install it with: sudo apt install mediainfo")

    if is_bt709:
        print("[Dispatch] Injecting color parameters.")
    else:
        print("[Dispatch] Using standard parameters.")

    # --- Construct Final Command ---
    # Use standard python3
    final_cmd = ["python3", av1an_script]

    # Parameters to inject if BT.709
    bt709_flags = (
        " --color-primaries 1 --transfer-characteristics 1 --matrix-coefficients 1"
    )

    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        # If we find the parameter strings, append flags if needed
        if arg in ("--fast-params", "--final-params"):
            final_cmd.append(arg)
            if idx + 1 < len(args):
                param_str = args[idx + 1]
                if is_bt709:
                    # Append our flags to the existing string
                    param_str += bt709_flags
                final_cmd.append(param_str)
                skip_next = True
            else:
                final_cmd.append("")
        else:
            final_cmd.append(arg)

    # --- Execute ---
    try:
        sys.stdout.flush()
        # On Linux, subprocess.check_call is safer for list args than manual joining.
        subprocess.check_call(final_cmd)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Could not execute {av1an_script}. Check file paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()
