package com.example.unetsegmentation

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import com.example.unetsegmentation.ui.theme.UNetSegmentationTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            UNetSegmentationTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    SegmentationScreen()
                }
            }
        }
    }
}

@Composable
fun SegmentationScreen() {
    val context = LocalContext.current
    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var maskBitmap     by remember { mutableStateOf<Bitmap?>(null) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val stream = context.contentResolver.openInputStream(it)
            originalBitmap = BitmapFactory.decodeStream(stream)
            maskBitmap = null
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Box(modifier = Modifier.weight(1f).fillMaxHeight()) {
                originalBitmap?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = "Imagem Original",
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
            Box(modifier = Modifier.weight(1f).fillMaxHeight()) {
                maskBitmap?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = "Máscara Segmentada",
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
        }

        Button(
            onClick = { launcher.launch("image/*") },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Selecionar Imagem")
        }

        Button(
            onClick = {
                originalBitmap?.let { bmp ->
                    maskBitmap = runSegmentation(context, bmp)
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Segmentar")
        }
    }
}

fun loadModelFile(context: android.content.Context): ByteBuffer {
    val fileDescriptor = context.assets.openFd("unet_segmentation.tflite")
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    return fileChannel.map(
        FileChannel.MapMode.READ_ONLY,
        fileDescriptor.startOffset,
        fileDescriptor.declaredLength
    )
}

fun runSegmentation(context: android.content.Context, bitmap: Bitmap): Bitmap {
    val imageSize = 128

    // Pré-processamento — NHWC (batch, altura, largura, canais)
    val resized = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
    val inputBuffer = ByteBuffer.allocateDirect(1 * imageSize * imageSize * 3 * 4)
    inputBuffer.order(ByteOrder.nativeOrder())

    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    for (y in 0 until imageSize) {
        for (x in 0 until imageSize) {
            val pixel = resized.getPixel(x, y)
            inputBuffer.putFloat((Color.red(pixel)   / 255f - mean[0]) / std[0])
            inputBuffer.putFloat((Color.green(pixel) / 255f - mean[1]) / std[1])
            inputBuffer.putFloat((Color.blue(pixel)  / 255f - mean[2]) / std[2])
        }
    }

    // Output — NHWC [1, 128, 128, 3]
    val outputBuffer = Array(1) { Array(imageSize) { Array(imageSize) { FloatArray(3) } } }

    val interpreter = Interpreter(loadModelFile(context))
    interpreter.run(inputBuffer, outputBuffer)
    interpreter.close()

    // Pós-processamento: argmax → cor
    val colors = intArrayOf(
        Color.rgb(255, 0, 0),   // 0 = Pet   → vermelho
        Color.rgb(0, 255, 0),   // 1 = Fundo → verde
        Color.rgb(0, 0, 255),   // 2 = Borda → azul
    )

    val maskPixels = IntArray(imageSize * imageSize)
    for (y in 0 until imageSize) {
        for (x in 0 until imageSize) {
            val scores = outputBuffer[0][y][x]
            val cls = scores.indices.maxByOrNull { scores[it] } ?: 0
            maskPixels[y * imageSize + x] = colors[cls]
        }
    }

    return Bitmap.createBitmap(maskPixels, imageSize, imageSize, Bitmap.Config.ARGB_8888)
}