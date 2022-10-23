# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Visualization
import plotly.express as px # for data visualization

# Skleran
from sklearn.datasets import make_swiss_roll # for creating swiss roll
from sklearn.manifold import LocallyLinearEmbedding as LLE # for LLE dimensionality reduction
from sklearn.manifold import Isomap # for Isomap dimensionality reduction

# Create a swiss roll
X, y = make_swiss_roll(n_samples=2000, noise=0.05)

# Make it thinner
X[:, 1] *= .5

# Create a flat addon to the top of the swiss roll
X_x=np.zeros((300,1))
X_y=np.random.uniform(low=0, high=10, size=(300,1))
X_z=np.random.uniform(low=14, high=25, size=(300,1))
X2=np.concatenate((X_x, X_y, X_z), axis=1)
y2=X_z.reshape(300)

# Concatenate swiss roll and flat rectangle arrays
X_two=np.concatenate((X, X2))
y_two=np.concatenate((y, y2))
# Create a 3D scatter plot
def Plot3D(X, y, plot_name):
    fig = px.scatter_3d(None, 
                        x=X[:,0], y=X[:,1], z=X[:,2],
                        color=y,
                        height=800, width=800
                       )
    # Update chart looks
    fig.update_layout(title_text=plot_name,
                      showlegend=False,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                            center=dict(x=0, y=0, z=-0.1),
                                            eye=dict(x=1.5, y=1.75, z=1)),
                                            margin=dict(l=0, r=0, b=0, t=0),
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             ),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, 
                                  line=dict(color='black', width=0.1)))
    fig.update(layout_coloraxis_showscale=False)
    fig.write_html( plot_name +'.html')
    return fig
    
    
#----------------------------------------------      
# Create a 2D scatter plot
def Plot2D(X, y, plot_name):
    # Create a scatter plot
    fig = px.scatter(None, x=X[:,0], y=X[:,1], 
                     color=y,
                     height=800, width=800
)


    fig.update_layout(title_text=plot_name,
                      showlegend=False,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             ),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                              )

                                            ))

    # Update marker size
    fig.update_traces(marker=dict(size=5,
                                 line=dict(color='black', width=1.3)))
    fig.update(layout_coloraxis_showscale=False)
    fig.write_html( plot_name +'.html')

    return fig

Plot3D(X, y, "1 Regular Swiss Roll")
Plot3D(X_two, y_two, "2 Modified Swiss Roll")
# Function for performing LLE and MLLE
def run_lle(num_neighbors, dims, mthd, data):
    # Specify LLE parameters
    embed_lle = LLE(n_neighbors=num_neighbors, # default=5, number of neighbors to consider for each point.
                    n_components=dims, # default=2, number of dimensions of the new space 
                    reg=0.001, # default=1e-3, regularization constant, multiplies the trace of the local covariance matrix of the distances.
                    eigen_solver='auto', # {‘auto’, ‘arpack’, ‘dense’}, default=’auto’, auto : algorithm will attempt to choose the best method for input data
                    #tol=1e-06, # default=1e-6, Tolerance for ‘arpack’ method. Not used if eigen_solver==’dense’.
                    #max_iter=100, # default=100, maximum number of iterations for the arpack solver. Not used if eigen_solver==’dense’.
                    method=mthd, # {‘standard’, ‘hessian’, ‘modified’, ‘ltsa’}, default=’standard’
                    #hessian_tol=0.0001, # default=1e-4, Tolerance for Hessian eigenmapping method. Only used if method == 'hessian'
                    modified_tol=1e-12, # default=1e-12, Tolerance for modified LLE method. Only used if method == 'modified'
                    neighbors_algorithm='auto', # {‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, default=’auto’, algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance
                    random_state=42, # default=None, Determines the random number generator when eigen_solver == ‘arpack’. Pass an int for reproducible results across multiple function calls.
                    n_jobs=-1 # default=None, The number of parallel jobs to run. -1 means using all processors.
                   )
    # Fit and transofrm the data
    result = embed_lle.fit_transform(data)
    
    # Return results
    return result

#----------------------------------------------   
# Function for performing Isomap
def run_isomap(num_neighbors, dims, data):
    # Specify Isomap parameters
    embed_isomap = Isomap(n_neighbors=num_neighbors, n_components=dims, n_jobs=-1)
    
    # Fit and transofrm the data
    result = embed_isomap.fit_transform(data)
    
    # Return results
    return result
######### Regular swiss roll #########

# Standard LLE on a regular swiss roll
std_lle_res=run_lle(num_neighbors=30, dims=2, mthd="standard", data=X)

# Hessian LLE on a regular swiss roll
hessian_res=run_lle(num_neighbors=30, dims=2, mthd='hessian', data=X)

# Modified LLE on a regular swiss roll
mlle_res=run_lle(num_neighbors=30, dims=2, mthd='modified', data=X)

# LTSA LLE on a regular swiss roll
ltsa_res=run_lle(num_neighbors=30, dims=2, mthd='ltsa', data=X)

# Isomap on a regular swiss roll
isomap_res=run_isomap(num_neighbors=30, dims=2, data=X)


######### Modified swiss roll #########

# Modified LLE on a modified swiss roll
mod_mlle_res=run_lle(num_neighbors=30, dims=2, mthd='modified', data=X_two)

# Isomap on a modified swiss roll
mod_isomap_res=run_isomap(num_neighbors=30, dims=2, data=X_two)
Plot2D(std_lle_res, y, "3 Regular Swiss Roll - LLE")
Plot2D(hessian_res, y, '4 Regular Swiss Roll - HLLE')
Plot2D(mlle_res, y, '5 Regular Swiss Roll - MLLE')
Plot2D(ltsa_res, y, '6 Regular Swiss Roll - LTSA')
Plot2D(isomap_res, y, '7 Regular Swiss Roll - Isomap')

Plot2D(mod_mlle_res, y_two, '8 Modified Swiss Roll - MLLE')
Plot2D(mod_isomap_res, y_two, '9 Modified Swiss Roll - Isomap')
