package com.google.ar.core.examples.java.common.fragment;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.util.Log;
import android.widget.ImageView;

public class ImageCoordinateConverter {
    // 静态变量来存储相对坐标
//    public static float relativeX;
//    public static float relativeY;

//    /**
//     * 将触控坐标转换为原始图片坐标
//     * @param imageView 图片容器
//     * @param bitmap 原始位图
//     * @param touchX X轴触控点
//     * @param touchY Y轴触控点
//     * @return 原始图片坐标点
//     */
//    public static PointF convertToBitmapCoordinates(ImageView imageView, Bitmap bitmap,
//                                                    float touchX, float touchY) {
//        // 获取图片绘制矩阵
//        Matrix matrix = imageView.getImageMatrix();
//        float[] matrixValues = new float[9];
//        matrix.getValues(matrixValues);
//
//        // 计算缩放比例
//        float scaleX = matrixValues[Matrix.MSCALE_X];
//        float scaleY = matrixValues[Matrix.MSCALE_Y];
//
//        // 计算图片实际显示区域
//        float drawableX = matrixValues[Matrix.MTRANS_X];
//        float drawableY = matrixValues[Matrix.MTRANS_Y];
//        float drawableWidth = imageView.getWidth() - 2 * drawableX;
//        float drawableHeight = imageView.getHeight() - 2 * drawableY;
//
//        // 计算点击位置相对于实际图片的位置
//         relativeX = touchX - drawableX;
//         relativeY = touchY - drawableY;
//
//        // 转换为原始图片坐标
//        float rawX = (relativeX / drawableWidth) * bitmap.getWidth();
//        float rawY = (relativeY / drawableHeight) * bitmap.getHeight();
//
//        // 添加调试日志
//        Log.d("CoordinateDebug", "ImageView尺寸: " + imageView.getWidth() + "x" + imageView.getHeight());
//        Log.d("CoordinateDebug", "Bitmap尺寸: " + bitmap.getWidth() + "x" + bitmap.getHeight());
//        Log.d("CoordinateDebug", "矩阵缩放系数: scaleX=" + scaleX + " scaleY=" + scaleY);
//        Log.d("CoordinateDebug", "触控点("+touchX+","+touchY+") → 转换后("+rawX+","+rawY+")");
//
//
//        return new PointF(rawX, rawY);
//    }
public static PointF convertToBitmapCoordinates(ImageView imageView, Bitmap bitmap,
                                                float touchX, float touchY) {
    Matrix matrix = imageView.getImageMatrix();
    float[] matrixValues = new float[9];
    matrix.getValues(matrixValues);

    // 获取关键变换参数
    float transX = matrixValues[Matrix.MTRANS_X];
    float transY = matrixValues[Matrix.MTRANS_Y];
    float scaleX = matrixValues[Matrix.MSCALE_X];
    float scaleY = matrixValues[Matrix.MSCALE_Y];

    // 计算有效显示区域
    float displayedWidth = bitmap.getWidth() * scaleX;
    float displayedHeight = bitmap.getHeight() * scaleY;

    // 边距补偿计算
    float paddingX = (imageView.getWidth() - displayedWidth) / 2;
    float paddingY = (imageView.getHeight() - displayedHeight) / 2;

    // 精确坐标转换
    float rawX = (touchX - transX ) / scaleX;
    float rawY = (touchY - transY - paddingY) / scaleY;

    return new PointF(
            Math.max(0, Math.min(rawX, bitmap.getWidth())),
            Math.max(0, Math.min(rawY, bitmap.getHeight()))
    );
}

}
