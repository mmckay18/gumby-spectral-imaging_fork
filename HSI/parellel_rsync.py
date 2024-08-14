import os
import subprocess
from multiprocessing import Pool, Manager
from tqdm import tqdm

# Define source and destination
SRC = "/gscratch/scrubbed/mmckay18/DATA/"
DST = "/scr/mmckay18/DATA/"

def generate_file_list(src):
    """Generate a list of all files in the source directory."""
    file_list = []
    for root, _, files in os.walk(src):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def copy_file(args):
    """Copy a single file using rsync."""
    file, progress_queue = args
    rel_path = os.path.relpath(file, SRC)
    dest_dir = os.path.join(DST, os.path.dirname(rel_path))
    os.makedirs(dest_dir, exist_ok=True)
    command = ["rsync", "-avz", "--progress", file, os.path.join(dest_dir, "")]
    subprocess.run(command)
    progress_queue.put(1)

def update_progress_bar(progress_queue, total_files):
    """Update the progress bar based on the progress queue."""
    pbar = tqdm(total=total_files)
    for _ in range(total_files):
        progress_queue.get()
        pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    # Generate a list of files to be copied
    files = generate_file_list(SRC)
    total_files = len(files)

    # Create a Manager to handle the progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Number of parallel processes
    num_processes = 128

    # Create a pool of workers and copy files in parallel
    with Pool(num_processes) as pool:
        # Start a process to update the progress bar
        pool.apply_async(update_progress_bar, (progress_queue, total_files))
        
        # Copy files in parallel
        pool.map(copy_file, [(file, progress_queue) for file in files])
