# Computer Vision - Clasificación de Reciclables

Repositorio para el trabajo práctico final de Computer Vision. Sistema de clasificación automática de residuos reciclables usando MobileNetV3 con integración Arduino para control físico.

## Descripción del Proyecto

Este proyecto implementa un clasificador de residuos que utiliza visión por computadora para identificar 4 categorías:
- **cardboard_paper**: Cartón y papel
- **ecoglasses**: Vasos ecológicos y eco-plásticos
- **metal_plastic**: Metal y plástico tradicional
- **trash**: Basura general

El sistema incluye:
- Modelo MobileNetV3 entrenado para clasificación
- Visualización Grad-CAM para interpretabilidad
- Inferencia en tiempo real con cámara
- Integración con Arduino para control de hardware (clasificación física)

## Estructura del Proyecto

### `/src` - Código fuente principal

#### `src/data/`
- **`dataset.py`**: Clases PyTorch Dataset para carga de datos
- **`preprocessing.py`**: Preprocesamiento y división train/val/test
- **`augmentation.py`**: Transformaciones y data augmentation

#### `src/models/`
- **`mobilenet.py`**: Arquitectura MobileNetV3 modificada para clasificación

#### `src/training/`
- **`trainer.py`**: Loop de entrenamiento principal
- **`losses.py`**: Funciones de pérdida (CrossEntropy, Focal Loss)
- **`metrics.py`**: Métricas de evaluación (accuracy, precision, recall, F1)
- **`callbacks.py`**: Early stopping y guardado de checkpoints

#### `src/inference/`
- **`predictor.py`**: Predictor para inferencia en imágenes individuales
- **`camera.py`**: Inferencia en tiempo real con cámara web
- **`gradcam.py`**: Implementación de Grad-CAM para visualización
- **`camera_gradcam.py`**: Cámara con visualización Grad-CAM integrada

#### `src/utils/`
- **`config.py`**: Gestión de configuraciones YAML
- **`logger.py`**: Sistema de logging
- **`visualization.py`**: Utilidades para gráficos y visualización

#### `src/arduino/`
- Código de referencia para Arduino Nano
- Controla servos para clasificación física de residuos en 4 cuadrantes
- Comunicación serial USB (9600 baud)
- **`arduino_tester.ino`**: Código de prueba para Arduino
- **`classification_tester.ino`**: Código de prueba de funcionamiento de coordinación de ambos servos
- **`serial_instructions.ino`**: Código principal para recibir instrucciones de clasificación desde el PC por serial y mover los servos
  
### `/configs`
- **`mobilenet_config.yaml`**: Hiperparámetros del modelo y entrenamiento

### `/notebooks`
- **`01_data_preparation.ipynb`**: Preparación y exploración de datos
- **`02_train_mobilenet.ipynb`**: Entrenamiento del modelo
- **`03_inference_demo.ipynb`**: **Notebook principal de inferencia** 

### `/data`
- **`raw/`**: Dataset original de clasificación de basura
- **`processed/`**: Datos procesados (train/val/test splits)
- **`custom/`**: Imágenes personalizadas para testing (ecoglasses)

### `/outputs`
- **`checkpoints/`**: Modelos entrenados (`.pt`)
- **`exports/`**: Modelos exportados (ONNX, TorchScript)
- **`logs/`**: Logs de entrenamiento

## Uso Rápido

### 1. Inferencia con Notebook (Recomendado)

Abrir y ejecutar **`notebooks/03_inference_demo.ipynb`**:

#### Contenido del notebook:

1. **Configuración inicial**: Importar módulos y cargar modelo entrenado
2. **ArduinoController**: Clase para comunicación serial con Arduino
   - Envía instrucciones (0-3) según la clase predicha
   - Configurable para Windows/Linux/Mac
3. **Carga del predictor**: Inicializa el modelo con pesos entrenados
4. **Predicción en imágenes estáticas**: Test con imágenes individuales
5. **Visualización Grad-CAM**: Mapas de calor mostrando regiones importantes
6. **Inferencia en tiempo real**: 
   - Cámara básica (`CameraInference`)
   - Cámara con Grad-CAM (`CameraInferenceWithGradCAM`)
   - Cámara con Arduino integrado (`CameraInferenceWithArduino`)
7. **Control manual del Arduino**: Testing de comunicación serial

#### Ejecutar inferencia con cámara:

```python
# Cámara normal (mono)
camera = CameraInferenceWithGradCAM(predictor, camera_id=0, enable_gradcam=True)
camera.run()

# Cámara ESTÉREO - usar solo vista izquierda o derecha
camera = CameraInferenceWithGradCAM(
    predictor, 
    camera_id=0, 
    enable_gradcam=True,
    stereo_mode='left'  # o 'right' para vista derecha
)
camera.run()

# Con Arduino y detección de bandeja vacía
arduino = ArduinoController(port='COM3', baudrate=9600)  # Ajustar puerto
camera_arduino = CameraInferenceWithArduino(
    predictor, 
    arduino, 
    camera_id=0,
    stability_duration=4.0,  # Espera 4 segundos de clasificación estable
    stereo_mode='left',  # Para cámara estéreo, usar 'left' o 'right'
    black_threshold=0.7,  # 70% de pixeles negros = bandeja vacía
    brightness_threshold=40  # Brillo máximo para considerar pixel "negro"
)
camera_arduino.run()
```

**Mejoras importantes:**
- **Estabilidad temporal**: El sistema espera 4-5 segundos con la misma clasificación antes de enviar al Arduino, evitando clasificaciones erróneas por frames individuales
- **Detección de bandeja vacía**: No clasifica cuando detecta que la bandeja está vacía (mayoría de pixeles negros), evitando movimientos del motor sin objeto
- **Zoom digital**: Acercar/alejar la imagen con teclas `+`/`-` (rango: 1.0x a 3.0x)
- **Soporte para cámara estéreo**: Extrae automáticamente la vista izquierda o derecha antes de procesar
- **Indicador visual**: Muestra "✓ STABLE" cuando la clasificación es consistente y "BANDEJA VACIA" cuando no hay objeto

### 2. Integración con Arduino

#### Hardware requerido:
- Arduino Nano (o compatible)
- 2 servomotores (control de clasificación física)
- Conexión USB al computador

#### Configuración:

1. Subir código de `src/arduino_instruction.ino` al Arduino usando Arduino IDE
2. Conectar Arduino por USB
3. Identificar puerto serial:
   - **Windows**: `COM3`, `COM4`, etc.
   - **Linux**: `/dev/ttyUSB0`, `/dev/ttyACM0`
   - **macOS**: `/dev/tty.usbserial-*`

4. En el notebook, configurar puerto correcto:
```python
arduino = ArduinoController(port='TU_PUERTO_AQUI', baudrate=9600)
```

#### Funcionamiento:
- El modelo predice la clase del residuo (0-3)
- Se envía el ID de clase al Arduino por serial
- El Arduino mueve los servos al cuadrante correspondiente
- El residuo se clasifica físicamente en el contenedor correcto

#### Mapeo de cuadrantes:
```
┌─────────┬─────────┐
│    0    │    1    │  0: cardboard_paper
│         │         │  1: ecoglasses
├─────────┼─────────┤  2: metal_plastic
│    2    │    3    │  3: trash
└─────────┴─────────┘
```

## Dependencias Principales

- PyTorch
- OpenCV (`cv2`)
- torchvision
- numpy, pandas
- matplotlib, seaborn
- pyserial (para Arduino)
- PIL/Pillow

## 🎮 Controles de Cámara

- **`q`**: Salir
- **`g`**: Toggle Grad-CAM (activar/desactivar visualización)
- **`s`**: Guardar frame actual
- **`+` / `=`**: Zoom in (acercar)
- **`-` / `_`**: Zoom out (alejar)

### Funcionamiento del Sistema de Estabilidad

Para evitar enviar múltiples comandos al Arduino en cada frame de video:

```
Frame 1: plastic (0.85) ─┐
Frame 2: plastic (0.88)  ├─── Acumulando...
Frame 3: plastic (0.82)  │
Frame 4: plastic (0.90)  ├─── 4 segundos ✓
Frame 5: plastic (0.87) ─┘    └─→ ENVIAR al Arduino (clase 2)
Frame 6: plastic (0.91) ────── No enviar (ya enviado)
Frame 7: glass (0.75)   ─┐
Frame 8: glass (0.80)    ├─── Acumulando nueva clase...
...
```

**Proceso:**
1. El sistema rastrea las últimas predicciones con alta confianza (>0.7)
2. Solo cuando **la misma clase se mantiene por 4-5 segundos consecutivos**, se considera "estable"
3. Una vez estable, se envía **una única instrucción** al Arduino
4. No se envía otra instrucción hasta que cambie la clasificación y se estabilice nuevamente

Esto previene movimientos erráticos del hardware y mejora la precisión del sistema físico.

## Resultados del Modelo

El modelo entrenado se encuentra en `outputs/checkpoints/best_model.pt` y alcanza alta precisión en la clasificación de las 4 categorías. Los resultados detallados de entrenamiento están en `PROJECT_REPORT.md`.

## Entrenamiento

Para reentrenar el modelo, ejecutar secuencialmente:
1. `notebooks/01_data_preparation.ipynb` - Preparar datos
2. `notebooks/02_train_mobilenet.ipynb` - Entrenar modelo

## 📝 Notas

- Los archivos `.py` en la raíz fueron utilizados para tests durante desarrollo
- El código de producción está en el directorio `src/`
- Grad-CAM ayuda a entender qué características visuales usa el modelo para clasificar
- El umbral de confianza para Arduino es configurable (default: 0.7)

### ⚡ Características Avanzadas

#### Sistema de Estabilidad Temporal
- Previene envíos múltiples al Arduino durante video continuo
- Requiere clasificación consistente durante 4-5 segundos antes de enviar
- Configurable mediante el parámetro `stability_duration`

#### Zoom Digital
- Acercamiento hasta 3x sin pérdida de calidad significativa
- Útil para objetos pequeños o distantes
- Control en tiempo real con teclas `+` y `-`
- Funciona correctamente con cámaras estéreo (aplicado después de extraer vista única)

#### Soporte Cámara Estéreo
- Detecta automáticamente si tu cámara envía dos vistas lado a lado
- Extrae solo la vista izquierda o derecha antes de procesar
- Evita problemas de zoom "lateral" en sistemas estéreo
- Configuración: `stereo_mode='left'` o `'right'` (None para cámara normal)

#### Detección de Bandeja Vacía
- Analiza el porcentaje de pixeles oscuros en cada frame
- No clasifica ni envía comandos cuando la bandeja está vacía (fondo negro)
- Previene movimientos innecesarios del motor
- Muestra "BANDEJA VACIA - Esperando objeto..." en pantalla
- Parámetros ajustables:
  - `black_threshold`: Porcentaje de negro para considerar vacío (default: 0.7 = 70%)
  - `brightness_threshold`: Brillo máximo para pixel "negro" (default: 40/255)

#### Visualización Mejorada
- Indicador de estabilidad en pantalla ("✓ STABLE")
- Color verde cuando hay clasificación estable
- Probabilidades de todas las clases en tiempo real