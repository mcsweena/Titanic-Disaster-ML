import pandas as pd


def gender_model():
    predictions = {}
    df = pd.read_csv('test.csv')

    for passenger_index, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']
        if passenger['Sex'] == 'male':
            predictions[passenger_id] = 0
        else:
            predictions[passenger_id] = 1

    print(predictions)
    new_df = pd.DataFrame.from_dict(predictions, orient='index')
    print(new_df.head())

    new_df.to_csv('genderbasedmodel_result.csv')


gender_model()
