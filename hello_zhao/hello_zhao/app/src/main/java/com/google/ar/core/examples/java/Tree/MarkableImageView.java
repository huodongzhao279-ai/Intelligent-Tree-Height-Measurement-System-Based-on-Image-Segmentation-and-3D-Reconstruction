package com.google.ar.core.examples.java.Tree;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.util.AttributeSet;

import androidx.appcompat.widget.AppCompatImageView;

import java.util.ArrayList;
import java.util.List;

public class MarkableImageView extends AppCompatImageView {

//    private Paint paint;
//    private List<PointF> marks;
//
//    public MarkableImageView(Context context) {
//        super(context);
//        init();
//    }
//
//    public MarkableImageView(Context context, AttributeSet attrs) {
//        super(context, attrs);
//        init();
//    }
//
//    public MarkableImageView(Context context, AttributeSet attrs, int defStyleAttr) {
//        super(context, attrs, defStyleAttr);
//        init();
//    }
//
//    private void init() {
//        paint = new Paint();
//        paint.setColor(Color.RED);
//        paint.setStyle(Paint.Style.FILL);
//        paint.setStrokeWidth(10);
//        marks = new ArrayList<>();
//    }
//
//    public void addMark(float x, float y) {
//        marks.add(new PointF(x, y));
//        invalidate();
//    }
//
//    @Override
//    protected void onDraw(Canvas canvas) {
//        super.onDraw(canvas);
//        for (PointF mark : marks) {
//            canvas.drawCircle(mark.x, mark.y, 10, paint);
//        }
//    }
//
//    public List<PointF> getMarks() {
//        return marks;
//    }
    private List<PointF> marks = new ArrayList<>(); // 存储多个坐标
    private Paint paint;
    private static final int MAX_MARKS = 2; // 最大标记点数

//    private float markX = -1;
//    private float markY = -1;
//    private Paint paint; // 提前声明并复用Paint对象

    public MarkableImageView(Context context) {
        super(context);
        init();
    }

    public MarkableImageView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public MarkableImageView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        // 初始化Paint对象
        paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.FILL);
    }

    public void setMarkPosition(float x, float y) {
        if (marks.size() >= 2) {
            marks.clear(); // 超过两个点时清空，重新开始
        }
        marks.add(new PointF(x, y));
        invalidate(); // 刷新视图
    }
    @Override
    public boolean performClick() {
        // 调用父类方法以触发点击事件
        return super.performClick();
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (PointF mark : marks) {
            // 使用提前创建的Paint对象绘制标记
            canvas.drawCircle(mark.x, mark.y, 10, paint);
        }
    }

}
