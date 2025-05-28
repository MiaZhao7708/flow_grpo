
def add_prefix_suffix(path):
    path=path.replace("/openseg_blob/","")
    path_prefix = 'https://openseg.blob.core.windows.net/openseg-aml/'
    # path_suffix = '?sp=racwdl&st=2025-05-19T07:44:32Z&se=2025-05-26T06:44:32Z&skoid=4f2fed16-4a8b-4ee0-ac96-30fae5bf0b82&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-05-19T07:44:32Z&ske=2025-05-26T06:44:32Z&sks=b&skv=2024-11-04&sv=2024-11-04&sr=c&sig=Q2zoITq4hogqK0tHKj8%2Fx7d2etGqKifNEzpjd1%2BKO8Y%3D'
    path_suffix = '?sp=racwdl&st=2025-05-28T07:17:27Z&se=2025-06-03T15:17:27Z&skoid=3d7a1f76-21da-4ecc-b4d3-2560f85b8c3c&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-05-28T07:17:27Z&ske=2025-06-03T15:17:27Z&sks=b&skv=2024-11-04&spr=https&sv=2024-11-04&sr=c&sig=lwacBD7XG18qvXjke0Y3v%2Bc8EPxFY7ByqWv2I03WV7w%3D'
    return f"{path_prefix}{path}{path_suffix}"

