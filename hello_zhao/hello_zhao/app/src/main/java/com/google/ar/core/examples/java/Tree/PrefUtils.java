package com.google.ar.core.examples.java.Tree;

import android.content.Context;
import android.content.SharedPreferences;

public class PrefUtils {
    private static final String PREFS_NAME = "AppConfig";
    private static final String KEY_IP_ADDRESS = "ip_address";

    // 保存IP地址
    public static void saveIpAddress(Context context, String ip) {
        SharedPreferences.Editor editor =
                context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit();
        editor.putString(KEY_IP_ADDRESS, ip);
        editor.apply();
    }

    // 获取IP地址
    public static String getIpAddress(Context context) {
        if (context == null) return "";
        SharedPreferences prefs =
                context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return prefs.getString(KEY_IP_ADDRESS, ""); // 默认返回空字符串
    }
}
