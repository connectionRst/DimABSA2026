# %%
import os
import zipfile

# %% [markdown]
# ### Step 7: Save prediction results

# %%
# create zip

# zip name is subtask folder name
zip_name = "all.zip"

# zip the folder
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for task in range(1,4):
        subtask = "subtask_" + str(task)
        try:
            for root, _, files_in_dir in os.walk(subtask):
                for file in files_in_dir:
                    full_path = os.path.join(root, file)
                    zf_path = os.path.relpath(full_path, ".")
                    zf.write(full_path, zf_path)
        except FileNotFoundError:
            pass


