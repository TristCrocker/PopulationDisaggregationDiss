import torch.nn as nn
import model
import torch
import train_validate
from data_processing import load_data

if __name__ == "__main__":
    data = load_data("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/covariates/district/all_features_districts.csv", 2)



    input_size = 0
    output_size = 1
    hidden_size = 4
    message_passing_count = 4
    drop_prob = 0.05
    learning_rate = 0.01

    model = model.GCN(input_size, output_size, hidden_size, message_passing_count, drop_prob)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
    train_validate.train_loop(200, model, data, optimizer, loss_fn)

    model.eval()

    #Predict admin3 

    # with torch.no_grad():
    #     admin_3_predictions = model()
