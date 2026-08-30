import geopandas as gpd

gdf = gpd.read_file("Glacier_1980_1990_2000_2010.shp")

print(gdf.shape)                    # rows, columns
print(gdf.columns.tolist())         # column names
print(gdf.drop(columns="geometry").head(10))
print(gdf.drop(columns="geometry").describe(include="all"))