# %%

from huggingface_hub import HfApi, upload_folder

api = HfApi()

# repo_id = "qwen3-8b-layer0-decoder-train-layers-9-18-27"
username = "adamkarvonen"


# repo_ids = [
#     # "checkpoints_all_pretrain_20_tokens_classification_posttrain",
#     "checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B",
#     "checkpoints_all_single_and_multi_pretrain_cls_posttrain_Qwen3-8B",
#     "checkpoints_all_single_and_multi_pretrain_latentqa_posttrain_Qwen3-8B",
# ]

# for repo_id in repo_ids:
#     folder = f"{repo_id}/final"

#     # create repo if it doesn't exist
#     api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

#     api.upload_folder(folder_path=folder, repo_id=f"{username}/{repo_id}", repo_type="model")

repo_ids = [
    # "checkpoints_all_pretrain_20_tokens_classification_posttrain",
    # "checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B",
    # "checkpoints_all_single_and_multi_pretrain_cls_posttrain_Qwen3-8B",
    "model_lora/gemma-2-9b-it-shuffled_3_epochs",
]

for repo_id in repo_ids:
    folder = f"{repo_id}"

    repo_id = repo_id.split("/")[-1]

    # create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_folder(folder_path=folder, repo_id=f"{username}/{repo_id}", repo_type="model")
