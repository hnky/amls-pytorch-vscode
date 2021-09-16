az config set defaults.workspace=Cybertron
az config set defaults.group=Cybertron-RG
az ml endpoint update --name simpsons-demo --deployment-file 04_Deploy/03_add_deployment.yml --traffic "v01:50,v02:50"