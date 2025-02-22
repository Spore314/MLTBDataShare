from pymongo import MongoClient  
import gridfs  
import os  
import shutil
import glob
import tarfile
import random  
from tqdm import tqdm
  
def connect_to_database(): 
    client = 
    db = client.tightbinding  
    fs = gridfs.GridFS(db, collection="tb_matrix") 
    fb = gridfs.GridFS(db, collection="tb_calfiles") 
    fn = db.tb_calfiles.files
    return fs, fb, fn
  
def get_cal_done_list(fn):  
    cal_done_list = []
    for doc in fn.find({}, {'struct_id': 1}):  
        cal_done_list.append(doc['struct_id'])
    return cal_done_list
  
def user_input(cal_done_list): 
    total_structures = len(cal_done_list) 
    user_input = input(f"Input the number of structures you want (Randomly selected from {total_structures} structures): ")  
    error_message = f"Invalid input: Expecting a positive integer N <= {total_structures}." 
    try:  
        N = int(user_input)  
        if N <= 0 or N > total_structures:  
            raise ValueError(error_message)  
        return N
    except ValueError:  
        print(error_message)  
        exit(1)  

def extract_files(folder_name, filename):
    section_markers = {  
        "STRU :": "STRU", 
        "TB matrix :": "data-HR-sparse_SPIN0.csr",  
        "SR matrix :": "data-SR-sparse_SPIN0.csr",
        "POSCAR :": "POSCAR",
        "KPT_SCF :": "KPT_SCF", 
        "INPUT :": "INPUT",
        "Input :": "Input" 
    }  
    current_section = None  
    section_content = ""  
  
    file_path = os.path.join(folder_name, filename)
    with open(file_path, 'r') as file:  
        for line in file:  
            line_strip = line.strip()  
            if line_strip in section_markers:  
                if current_section:  
                    with open(os.path.join(folder_name, section_markers[current_section.strip()]), 'w') as f:  
                        f.write(section_content.strip())  
                    section_content = ""  
                current_section = line  
            elif current_section:
                if line.startswith("E-Fermi"):
                    with open(os.path.join(folder_name, 'E-Fermi'), 'w') as f:  
                        f.write(line)  
                else:
                    section_content += line 
  
    # Write the last section
    if current_section and section_content:  
        with open(os.path.join(folder_name, section_markers[current_section.strip()]), 'w') as f:  
            f.write(section_content.strip())  

def download_files(fs, fb, cal_done_list, N):  
    selected_struct_ids = random.sample(cal_done_list, N)
    folder_format = "{:0" + str(len(str(N))) + "d}"
    for i, struct_id in tqdm(enumerate(selected_struct_ids, 1), total=N):
    #for i, struct_id in enumerate(selected_struct_ids, 1):
        folder_name = folder_format.format(i)
        os.makedirs(folder_name, exist_ok=True)

        # Write struct_id
        id_path = os.path.join(folder_name, 'struct_id')
        with open(id_path, 'w') as f:
            f.write(struct_id)

        # Download files: STRU-HR-SR
        out = fs.get_last_version(filename=struct_id)
        file_path = os.path.join(folder_name, 'data-STRU-HR-download')
        with open(file_path, 'wb') as f:
            f.write(out.read())
        extract_files(folder_name, 'data-STRU-HR-download')
        os.remove(os.path.join(folder_name, 'STRU')) # Get STRU in Str_id-*.tar.gz

        # Download files: Str_id-*.tar.gz
        out = fb.get_last_version(struct_id=struct_id)
        with open('STRU.tar.gz', 'wb') as f:
            f.write(out.read())
        with tarfile.open('STRU.tar.gz', 'r:gz') as tar:
            tar.extractall()  # will not overwrite existing files

        base_name = glob.glob('Str_id-*')[0]
        files_need = glob.glob(os.path.join(base_name, '*'))
        for file_need in files_need:
            shutil.move(file_need, folder_name)
        shutil.rmtree(base_name)
        os.remove('STRU.tar.gz')

        # Remove tmp files: data-STRU-HR-download
        os.remove(os.path.join(folder_name, 'data-STRU-HR-download'))
 
def main():  
    fs, fb, fn = connect_to_database()           # Connect to the database 
    cal_done_list = get_cal_done_list(fn)        # Collect all 'struct_id' from the database  
    N = user_input(cal_done_list)                # Get the number of structures to download 
    download_files(fs, fb, cal_done_list, N)     # Download the selected structures       
    print("All structures have been downloaded successfully.")  
  
if __name__ == "__main__":  
    main()
