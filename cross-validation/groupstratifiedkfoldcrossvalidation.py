import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    data_dir = "../data/"
    processed_dir = "../processed/"

    train_data = pd.read_csv(f"{data_dir}melanoma_train.csv")

    randomized_data= train_data.sample(frac=1).reset_index(drop=True)

    print(f"patient_id distribution: {randomized_data['patient_id'].value_counts()}")

    #initializing groupkfold instance
    gkf = model_selection.GroupKFold(n_splits=5)

    # In the dataset the patient_id is not unique, based on this we need to mention what column values goes in as groups.
    y = randomized_data.target.values
    groups= randomized_data.patient_id.values
    for f,(t,v) in enumerate(gkf.split(X=randomized_data, y =y , groups=groups)):
        randomized_data.loc[v,'gkfold'] = f
    print(f"gfk_value_1 distribution: {randomized_data[randomized_data['gkfold'] == 1]['patient_id'].value_counts()}")
    
    # After this step, we observe that the distribution is not proportionate according to patient_id. Thus implimenting stratified kfold cross-validation on gkfold column
    randomized_data['kfold'] = -1

    #Initiating the stratified kfold instance
    skf = model_selection.StratifiedKFold(n_splits = 5)

    for f, (t,v) in enumerate(skf.split(X= randomized_data, y = randomized_data.gkfold.values)):
        randomized_data.loc[v,'kfold'] = f
    
    randomized_data.to_csv(f"{processed_dir}groupkfold_melanoma.csv",index=False)
    print(f"gfk_value_1 after stratified kfold cross validatio:\n:{randomized_data[randomized_data['kfold'] == 1]['patient_id'].value_counts()}")
    