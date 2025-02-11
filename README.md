# rooftop

Epoch 1/30
47/47 ━━━━━━━━━━━━━━━━━━━━ 274s 6s/step - loss: 0.7427 - mean_io_u_1: 0.4741 - val_loss: 0.7260 - val_mean_io_u_1: 0.4752 - learning_rate: 1.0000e-04
Epoch 2/30
47/47 ━━━━━━━━━━━━━━━━━━━━ 261s 6s/step - loss: 0.4956 - mean_io_u_1: 0.4758 - val_loss: 0.4394 - val_mean_io_u_1: 0.4752 - learning_rate: 1.0000e-04


sudo apt update && sudo apt upgrade -y
sudo apt install update-manager-core
sudo nano /etc/update-manager/release-upgrades
sudo do-release-upgrade -d



sir gave me village data for rooftop segementation. It has ony 1 big image of village, Marhara, of 800mb and having 23351 × 22361 pixels. along with this image it has its annotaions, but I am not understanding this. Annotations has differernt files ending with, .cpg, .dbf, .prj, .sbn, .sbx, .shp, .shx. how to read this data to train my model, and also which files I needed for training, there are Electric_pole, Marhara_all_layer, Marhara_village, potential_area, RCC, Road, Road_center, Settlement_extent, Solar, tree, VAcant_space, Waterbodies. each of them has file ending with these extensions as mentioned above.
