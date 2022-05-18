Este archivo tiene como propósito explicar el paso a paso de cómo exportar el archivo csv de la memoria de la raspberry pi 2 (RPI2) y acceder a este hacia el ordenador al que se conecta la RPI2, mediante protocolo de comunicación SSH.

Para esto se deben seguir los siguientes pasos:
  1. Conectar la RPI2 a la computadora host que tenga el puerto Ethernet; debe tener acceso a internet.
  2. Encender la RPI2.
  3. En el menú de la RPI2, ejecutar python 3.
  4. Correr la aplicación en la RPI2 con el comando "nombre_de_la_app.py", sin las comillas.
  5. A continuación, deberá ejecutar Windows PowerShell.
  6. Luego tendrá que ejecutar el comando de la siguiente línea:
  
      scp -r root@169.254.61.80:/usr/bin/Data_tflite.csv /C:"directorio del archivo".
  
  Este comando creará una copia en la computadora del csv existente (al momento de su ejecucción) que se encuentra en la RPI2, con esto ya tendrá el          documento listo para abrir desde unos clics en los archivos de su máquina nativa.
  
