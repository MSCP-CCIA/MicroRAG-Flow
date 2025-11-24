#######################FRONTEND############
# Ver logs en tiempo real
sudo journalctl -u streamlit.service -f

# Detener el servicio
sudo systemctl stop streamlit.service

# Reiniciar el servicio
sudo systemctl restart streamlit.service

# Deshabilitar el autoarranque
sudo systemctl disable streamlit.service

######################CHROMA-DB###########################################
# Ver logs en tiempo real
sudo journalctl -u chroma.service -f

# Ver últimas 50 líneas de logs
sudo journalctl -u chroma.service -n 50

# Reiniciar el servicio
sudo systemctl restart chroma.service

# Detener el servicio
sudo systemctl stop chroma.service


####################ORQUESTADOR##################################
# Reiniciar todos los servicios
sudo systemctl restart orchestrator.service

# Detener todos
sudo systemctl stop orchestrator.service

# Ver estado de todos
sudo systemctl status orchestrator.service