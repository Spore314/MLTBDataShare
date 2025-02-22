**Quick Instructions**:

1. **Set MongoDB Connection**:  
   In `get_hr.py`, update `MongoClient("mongodb://")` with your server info.

2. **Get Data**:  
   Run `python get_hr.py` to fetch data with IDs.  
   Or Run `python get_calfiles.py` to fetch more calculation files with IDs.

4. **Create NPZ File**:  
   Execute `python MLTB_data.py` to generate `MLTB_data.npz`.

5. **Check Data**:  
   Use `python read_npz.py` to verify data loading.
