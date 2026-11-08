import wandb

run = wandb.init(project="rag-eval-lab",config={"seed":12},name ="learning_artifact_exp_1")
for i in range(5):
    run.log({"fake_score": i * 0.1})

table = wandb.Table(columns = [
                                "question","answer"
                            ],
                    data = [
                                ["Q1?",     "A1"],
                                ["Q2?",     "A2"],
                                ["Q3?",     "A3"]
                            ]       
                    )
run.log({"fake_qa" : table})


artifact = wandb.Artifact(name = "fake_artifact", type = "dataset")
artifact.add_file(local_path="sandbox/fake_dataset.txt")
run.log_artifact(artifact)
run.finish()

