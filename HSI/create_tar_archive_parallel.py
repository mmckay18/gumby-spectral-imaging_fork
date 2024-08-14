import os
import subprocess
from multiprocessing import Pool, cpu_count

def create_sub_tar(sub_directory):
    try:
        tar_file = f"{sub_directory}.tar"
        # Create a tar archive for the subdirectory
        command = ['tar', '-cvf', tar_file, '-C', sub_directory, '.']
        subprocess.run(command, check=True)
        return tar_file
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while creating the tar archive for {sub_directory}: {e}")
        return None

def combine_tars(tar_files, output_tar_file):
    try:
        # Combine all sub tar files into one
        command = ['tar', '-cvf', output_tar_file] + tar_files
        subprocess.run(command, check=True)
        # Clean up sub tar files
        for tar_file in tar_files:
            os.remove(tar_file)
        print(f"Successfully created combined tar archive: {output_tar_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while combining tar archives: {e}")

def main(source_directory, output_tar_file):
    # Get list of subdirectories
    subdirectories = [os.path.join(source_directory, sub_dir) for sub_dir in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, sub_dir))]

    # Use multiprocessing to create tar archives for subdirectories in parallel
#     num_cores = cpu_count()  # Number of available CPU cores
    num_cores = 20
    print(f"Using {num_cores} cores for parallel processing.")
    
    with Pool(processes=num_cores) as pool:
        tar_files = pool.map(create_sub_tar, subdirectories)

    # Filter out any None values that might have occurred due to errors
    tar_files = [tar_file for tar_file in tar_files if tar_file]

    # Combine sub tar files into the final tar archive
    combine_tars(tar_files, output_tar_file)

if __name__ == "__main__":
    source_directory = "/gscratch/scrubbed/mmckay18/DATA"
    output_tar_file = "job_data.tar"

    main(source_directory, output_tar_file)
