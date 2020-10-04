package org.pytorch.helloworld;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

public class MainActivity extends AppCompatActivity {
  public static final int PICK_IMAGE = 1;
  Uri selectedImage;
  Module module = null;
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Button btnSelect = findViewById(R.id.buttonSelect);
    btnSelect.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        try {
          if (ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, PICK_IMAGE);
          } else {
            pickFromGallery();

          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    });
//    Bitmap bitmap = null;
//
//    try {
//      // creating bitmap from packaged into app android asset 'image.jpg',
//      // app/src/main/assets/image.jpg
//      bitmap = BitmapFactory.decodeStream(getAssets().open("kite.jpeg"));
//      // loading serialized torchscript module from packaged into app android asset model.pt,
//      // app/src/model/assets/model.pt
//
//    } catch (IOException e) {
//      Log.e("PytorchHelloWorld", "Error reading assets", e);
//      finish();
//    }
//
//    // showing image on UI
//    ImageView imageView = findViewById(R.id.image);
//    imageView.setImageBitmap(bitmap);

    // preparing input tensor

  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
  private void pickFromGallery() {

    Intent intent = new Intent(Intent.ACTION_PICK);
    intent.setType("image/*");
    String[] mimeTypes = {"image/jpeg", "image/png"};
    intent.putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes);
    startActivityForResult(intent, PICK_IMAGE);
  }
  @Override
  public void onActivityResult(int requestCode, int resultCode, Intent data) {


    super.onActivityResult(requestCode, resultCode, data);
    if (resultCode == MainActivity.RESULT_OK)
      switch (requestCode) {
        case PICK_IMAGE:
          try {
            module = Module.load(assetFilePath(this, "model.pt"));
          } catch (IOException e) {
            e.printStackTrace();
          }
          selectedImage = data.getData();
          ImageView imageView = findViewById(R.id.image);
          imageView.setImageURI(selectedImage);

          Bitmap bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
          bitmap=Bitmap.createScaledBitmap(bitmap, 320, 320, false);
          final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                  TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);


          // running the model
          final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

          // getting tensor content as java array of floats
          final float[] scores = outputTensor.getDataAsFloatArray();

          // searching for the index with maximum score
          float maxScore = -Float.MAX_VALUE;
          int maxScoreIdx = -1;
          for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
              maxScore = scores[i];
              maxScoreIdx = i;
            }
          }

          String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

          // showing className on UI
          TextView textView = findViewById(R.id.text);
        textView.setText(className);

          break;
      }

  }
}
