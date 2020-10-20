az login --service-principal -u 144f73e6-2a64-47bc-8cfd-f6cbf04d9a69 -p jTn5Cz_b6qFQ-XF.eD9JoLQXSKbH4VwMq. --tenant e81f979e-21ba-433a-9fe1-673ae9641a5f
az disk create -g PyTorchCiTestGroup -n pytorch-test-vm-os-disk --source pytorch-ci-test-base-image --location SouthCentralUS
az vm create --name jozh-pytorch-test-vm-01 -g PyTorchCiTestGroup --attach-os-disk pytorch-test-vm-os-disk --location SouthCentralUS --os-type Windows
az vm run-command invoke  --command-id RunPowerShellScript --name jozh-pytorch-test-vm-01 -g PyTorchCiTestGroup --scripts "@script.ps1"  -o json
az vm delete --name jozh-pytorch-test-vm-01 -g PyTorchCiTestGroup -y
az disk delete --name pytorch-test-vm-os-disk -g PyTorchCiTestGroup -y