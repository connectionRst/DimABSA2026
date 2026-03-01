# %%
import os
import zipfile

#task config
subtask = "subtask_2" # subtask_2 or subtask_3


# %% [markdown]
# ### Step 7: Save prediction results

# %%
# create zip

# zip name is subtask folder name
zip_name = f"{subtask}.zip"

# zip the folder
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files_in_dir in os.walk(subtask):
        for file in files_in_dir:
            full_path = os.path.join(root, file)
            zf_path = os.path.relpath(full_path, ".")
            zf.write(full_path, zf_path)


