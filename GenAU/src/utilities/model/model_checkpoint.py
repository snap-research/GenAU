import os
import boto3
from pytorch_lightning.callbacks import ModelCheckpoint
import glob

class S3ModelCheckpoint(ModelCheckpoint):
    def __init__(self, bucket_name, s3_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.s3_folder = s3_folder
        if self.bucket_name is not None:
            self.s3_client = boto3.client('s3')
        self.last_checkpoint_path = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save the checkpoint locally as usual
        filepath = self.last_model_path
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if pl_module.global_rank == 0 and self.bucket_name is not None:
            list_of_files = glob.glob(os.path.join(self.dirpath, '*.ckpt'))  # Get list of all .ckpt files
            if list_of_files:
                filepath = max(list_of_files, key=os.path.getctime)  # Find the most recently created file

                # If this isn't the last checkpoint or it's not one of the top_k, upload and delete
                if filepath != self.last_checkpoint_path and not self._is_last_checkpoint(filepath):
                    self.upload_to_s3(filepath)
                    os.remove(filepath)
                else:
                    # Update the path of the last saved checkpoint
                    self.last_checkpoint_path = filepath

    def upload_to_s3(self, filepath):
        s3_path = os.path.join(self.s3_folder, filepath)
        self.s3_client.upload_file(filepath, self.bucket_name, s3_path)
        print(f"[INFO] Model checkpoint uploaded to {self.bucket_name}/{s3_path}")

    def _is_last_checkpoint(self, filepath):
        # Determine if this is the last checkpoint based on file naming
        # Assuming your last checkpoint is named like 'last.ckpt'
        return 'last' in filepath