import torch.nn as nn
import model
import torch
import train_validate
from data_processing import load_data

if __name__ == "__main__":
    data = load_data("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", "data/covariates/district/all_features_districts.csv", "data/covariates/postos/all_features_postos.csv", "data/mappings/mozam_admin_2_to_3_mappings.csv", 2, 3)
    num_features = data.x.shape[1]
   
    input_size = num_features
    output_size = 1
    hidden_size = 32
    message_passing_count = 6
    drop_prob = 0.2
    learning_rate = 1e-4

    GCN_model = model.GCN(input_size, output_size, hidden_size, message_passing_count, drop_prob)

    loss_fn = nn.SmoothL1Loss() 
    optimizer = torch.optim.Adam(GCN_model.parameters(), lr = learning_rate, weight_decay = 5e-4)
    train_validate.train_loop(300, GCN_model, data, optimizer, loss_fn)

    GCN_model.eval()

    #Predict admin3 
    # with torch.no_grad():
    #     admin_3_predictions = model()



