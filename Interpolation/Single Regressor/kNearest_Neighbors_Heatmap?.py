from sklearn.preprocessing import StandardScaler
import pandas
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import cv2

def simpleHeatMap(predict, mapData, display):
    np.random.seed(22)

    npvariables = StandardScaler().fit(mapData[['ProfCurv', 'Slope', 'TWI']].to_numpy()).transform(mapData[['ProfCurv', 'Slope', 'TWI']].to_numpy())
    npvariables = np.nan_to_num(npvariables, nan=0)

    xy = mapData[["POINT_X", "POINT_Y"]].to_numpy()

    scaledX = np.ceil((xy[:,0] - min(xy[:,0]))/9.4)
    scaledY = np.ceil((xy[:,1] - min(xy[:,1]))/9.4)
    scaledCoord = np.array([scaledX, scaledY]).T

    pixels = []

    for i in range(0, len(scaledCoord)):
        prediction = np.squeeze(predict(np.array([npvariables[i]])))
        print(prediction)
        pixels.append(prediction)

    heatmap = []

    for x in range(int(scaledX.min()), int(scaledX.max())):
        print(x)
        row = []
        for y in range(int(scaledY.min()), int(scaledY.max())):
            try:
                index = scaledCoord.tolist().index([x,y])
                row.append(pixels[index])
            except:
                row.append(np.NaN)
        heatmap.append(row)
    
    imageio.imwrite('output.tif', np.array(heatmap))
    
    # fig, ax = plt.subplots()
    # cmap = plt.cm.get_cmap('coolwarm') 
    # heatmap_img = ax.imshow(np.array(heatmap), cmap=cmap)
    # ax.axis('off')
    # output_path = 'heatmap.png'
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    if(display):
        plt.axis('on')
        plt.imshow(np.array(heatmap))
        plt.colorbar()
        plt.title("VWC")
        plt.show()


# def overlayHeatMap():
#     # Load the background image
#     background_image = cv2.imread('SataliteField.PNG')

#     # Load the overlay image
#     overlay_image = cv2.imread('heatmap.png')

#     # Resize the overlay image to match the dimensions of the background image
#     resized_overlay = cv2.resize(overlay_image, (background_image.shape[1], background_image.shape[0]))
#     height, width = resized_overlay.shape[:2]
#     center = (width / 2, height / 2)
#     angle = -3
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     resized_overlay = cv2.warpAffine(resized_overlay, rotation_matrix, (width, height))

#     # Set the overlay opacity
#     opacity = 0.3

#     # Perform the overlay operation
#     result = cv2.addWeighted(background_image, 1 - opacity, resized_overlay, opacity, 0)

#     # Save the resulting image
#     cv2.imwrite('output.png', result)

# overlayHeatMap()