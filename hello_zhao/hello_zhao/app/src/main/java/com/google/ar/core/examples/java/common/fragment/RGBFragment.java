package com.google.ar.core.examples.java.common.fragment;

import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import com.google.ar.core.examples.java.Tree.R;
import com.google.ar.core.examples.java.common.helpers.Constants;
import com.google.ar.core.examples.java.common.helpers.DatabaseHelper;
import com.google.ar.core.examples.java.common.helpers.ImageHelper;

import java.io.File;

import butterknife.ButterKnife;
import butterknife.OnClick;
import butterknife.Unbinder;
import com.google.ar.core.examples.java.Tree.MarkableImageView;
import com.google.ar.core.examples.java.Tree.R;
import com.google.ar.core.examples.java.Tree.ItemDetailActivity;

public class RGBFragment extends BaseFragment {

    public static final String BUNDLE_TITLE = "title";

    private View mContentView;
    private Unbinder unbinder;
    private Long id;
    private String imagePath;
    private Bitmap ImageBitmap=null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        mContext = getActivity();
        mContentView = inflater.inflate(R.layout.fragment_rgb_layout, container, false);
        initData();
        initView();
        return mContentView;
    }

    private void initView() {
        unbinder = ButterKnife.bind(this, mContentView);
        ImageView image=mContentView.findViewById(R.id.image);
        if(ImageBitmap!=null){
            image.setImageBitmap(ImageBitmap);
            image.setBackgroundResource(0);
            // 添加点击事件监听器
            image.setOnClickListener(v -> {
                // 空实现，改用OnTouchListener获取精确坐标
            });

            image.setOnTouchListener(new View.OnTouchListener() {
                @Override
                public boolean onTouch(View v, MotionEvent event) {
                    if (event.getAction() == MotionEvent.ACTION_DOWN) {
                        float touchX = event.getX();
                        float touchY = event.getY();
//                        System.out.println("TouchEvent: (" + touchX + ", " + touchY + ")");

                        // 后续转换逻辑
                        handleRealCoordinates(v, touchX, touchY);
                    }
                    return true; // 消费事件
                }
            });
        }
        else image.setBackgroundResource(R.drawable.noimage);
    }

    private void handleRealCoordinates(View view, float touchX, float touchY) {
        if (view instanceof MarkableImageView && ImageBitmap != null) {
            MarkableImageView markableView = (MarkableImageView) view;

            // 使用工具类转换坐标
            PointF bitmapPoint = ImageCoordinateConverter.convertToBitmapCoordinates(
                    markableView, ImageBitmap, touchX, touchY
            );

            // 更新标记位置
            markableView.setMarkPosition(touchX, touchY);
            markableView.invalidate();

            // 打印调试日志
            Log.d("TouchEvent", String.format("Converted coordinates: (%.1f, %.1f)",
                    bitmapPoint.x, bitmapPoint.y));

            // 传递坐标到 ItemDetailActivity
            ((ItemDetailActivity) getActivity()).setTouchCoordinates(bitmapPoint.x, bitmapPoint.y);

        }
    }


    //初始化数据
    private void initData() {
        Bundle bundle = new Bundle();
        bundle = this.getArguments();
        id = bundle.getLong("id");
        DatabaseHelper TreeDBHelper = new DatabaseHelper(mContext, Constants.TREE_TABLE_NAME+".db",null,1);
        SQLiteDatabase TreeDB = TreeDBHelper.getReadableDatabase();
        Cursor treeCursor = TreeDB.rawQuery("select image from "+Constants.TREE_TABLE_NAME+" where id="+id, null);

        while (treeCursor.moveToNext()){
            imagePath=treeCursor.getString(0);
        }
        File image = new File(imagePath);
        if(image.exists()){
            ImageBitmap = ImageHelper.getCompressedBitmap(1080,1440, image);
        }
    }

//    private float lastClickX = -1;
//    private float lastClickY = -1;
//
//    private void handleImageClick(View view, float x, float y) {
//        if (view instanceof MarkableImageView) {
//            MarkableImageView markableImageView = (MarkableImageView) view;
//
//            // 更新点击位置
//            lastClickX = x;
//            lastClickY = y;
//
//            // 打印或处理点击位置
//            System.out.println("Clicked at: (" + x + ", " + y + ")");
//
//            // 刷新点击位置（例如，可以在图片上绘制一个标记）
//            markableImageView.setMarkPosition(x, y);
//            markableImageView.invalidate(); // 刷新视图
//        }
//
//    }


    @OnClick({})
    public void onClick(View v) {
        switch (v.getId()) {
            default:
                break;
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        unbinder.unbind();
    }

    public static RGBFragment newInstance(String title) {
        Bundle bundle = new Bundle();
        bundle.putString(BUNDLE_TITLE, title);
        RGBFragment fragment = new RGBFragment();
        fragment.setArguments(bundle);
        return fragment;
    }

}