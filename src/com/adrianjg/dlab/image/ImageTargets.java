/*===============================================================================

Copyright (c) 2012-2014 Qualcomm Connected Experiences, Inc. All Rights Reserved.

Vuforia is a trademark of QUALCOMM Incorporated, registered in the United States 
and other countries. Trademarks of QUALCOMM Incorporated are used with permission.
===============================================================================*/

package com.adrianjg.dlab.image;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.Point;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.Display;
import android.view.GestureDetector;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup.LayoutParams;
import android.view.animation.AnimationUtils;
import android.view.animation.RotateAnimation;
import android.widget.AbsoluteLayout;
import android.widget.CheckBox;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.qualcomm.vuforia.CameraDevice;
import com.qualcomm.vuforia.DataSet;
import com.qualcomm.vuforia.ImageTracker;
import com.qualcomm.vuforia.State;
import com.qualcomm.vuforia.STORAGE_TYPE;
import com.qualcomm.vuforia.Trackable;
import com.qualcomm.vuforia.Tracker;
import com.qualcomm.vuforia.TrackerManager;
import com.qualcomm.vuforia.Vec2F;
import com.qualcomm.vuforia.VideoBackgroundConfig;
import com.qualcomm.vuforia.VideoMode;
import com.qualcomm.vuforia.Vuforia;
import com.adrianjg.dlab.ApplicationControl;
import com.adrianjg.dlab.ApplicationException;
import com.adrianjg.dlab.ApplicationSession;
import com.adrianjg.dlab.utils.LoadingDialogHandler;
import com.adrianjg.dlab.utils.ApplicationGLView;
import com.adrianjg.dlab.utils.Texture;
import com.adrianjg.dlab.R;
import com.adrianjg.dlab.ui.AppMenu;
import com.adrianjg.dlab.ui.AppMenuGroup;
import com.adrianjg.dlab.ui.AppMenuInterface;


public class ImageTargets extends Activity implements ApplicationControl,
    AppMenuInterface
{
    private static final String LOGTAG = "ImageTargets";
	private static final String AR_FOLDER = "AR Flow Meter";
	private static final String AR_FILE_EXT = ".txt";
	
	public final static String EXTRA_PATIENT_NAME = "com.example.respiratorydiseaseaudioanalyser.PATIENT_NAME";
	public final static String EXTRA_PATIENT_ID = "com.example.respiratorydiseaseaudioanalyser.PATIENT_ID";

    
    ApplicationSession vuforiaAppSession;
    
    private DataSet mCurrentDataset;
    private int mCurrentDatasetSelectionIndex = 0;
    private int mStartDatasetsIndex = 0;
    private int mDatasetsNumber = 0;
    private int onScreen = -1;
    private ArrayList<String> mDatasetStrings = new ArrayList<String>();
    
    // Our OpenGL view:
    private ApplicationGLView mGlView;
    
    // Our renderer:
    private ImageTargetRenderer mRenderer;
    private GestureDetector mGestureDetector;
    
    // The textures we will use for rendering:
    private Vector<Texture> mTextures;
    
    private boolean mSwitchDatasetAsap = false;
    private boolean mFlash = false;
    private boolean mContAutofocus = false;
    private boolean mExtendedTracking = false;
    
    private View mFlashOptionView;
    private RelativeLayout mUILayout;
    private RelativeLayout mOverlayView;
    private AppMenu mSampleAppMenu;
    LoadingDialogHandler loadingDialogHandler = new LoadingDialogHandler(this);
    
    boolean mIsDroidDevice = false;
	
    // Called when the activity first starts or the user navigates back to an
    // activity.
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        Log.d(LOGTAG, "onCreate");
        super.onCreate(savedInstanceState);
        
        vuforiaAppSession = new ApplicationSession(this);
   
        startLoadingAnimation();
        mDatasetStrings.add("DLab_marker.xml");
        
        vuforiaAppSession
            .initAR(this, ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        
        mGestureDetector = new GestureDetector(this, new GestureListener());
        
        // Load any sample specific textures:
        mTextures = new Vector<Texture>();
        loadTextures();
        
        mIsDroidDevice = android.os.Build.MODEL.toLowerCase().startsWith(
            "droid");
    }
    
    // Process Single Tap event to trigger autofocus
    private class GestureListener extends
        GestureDetector.SimpleOnGestureListener
    {
        // Used to set autofocus one second after a manual focus is triggered
        private final Handler autofocusHandler = new Handler();
        
        
        @Override
        public boolean onDown(MotionEvent e)
        {
            return true;
        }
        
        
        @Override
        public boolean onSingleTapUp(MotionEvent e)
        {
            // Generates a Handler to trigger autofocus
            // after 1 second
            autofocusHandler.postDelayed(new Runnable()
            {
                public void run()
                {
                    boolean result = CameraDevice.getInstance().setFocusMode(
                        CameraDevice.FOCUS_MODE.FOCUS_MODE_TRIGGERAUTO);
                    
                    if (!result)
                        Log.e("SingleTapUp", "Unable to trigger focus");
                }
            }, 1000L);
            
            return true;
        }
    }
    
    
    // We want to load specific textures from the APK, which we will later use
    // for rendering.
    
    private void loadTextures()
    {
    	mTextures.add(Texture.loadTextureFromApk("heart-tex.png", 
    		getAssets()));
    }
       
    // Called when the activity will start interacting with the user.
    @Override
    protected void onResume()
    {
        Log.d(LOGTAG, "onResume");
        super.onResume();
        
        // This is needed for some Droid devices to force portrait
        if (mIsDroidDevice)
        {
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        }
        
        try
        {
            vuforiaAppSession.resumeAR();
        } catch (ApplicationException e)
        {
            Log.e(LOGTAG, e.getString());
        }
        
        // Resume the GL view:
        if (mGlView != null)
        {
            mGlView.setVisibility(View.VISIBLE);
            mGlView.onResume();
        }
        
    }
    
    
    // Callback for configuration changes the activity handles itself
    @Override
    public void onConfigurationChanged(Configuration config)
    {
        Log.d(LOGTAG, "onConfigurationChanged");
        super.onConfigurationChanged(config);
        
        vuforiaAppSession.onConfigurationChanged();
    }
    
    
    // Called when the system is about to start resuming a previous activity.
    @Override
    protected void onPause()
    {
        Log.d(LOGTAG, "onPause");
        super.onPause();
        
        if (mGlView != null)
        {
            mGlView.setVisibility(View.INVISIBLE);
            mGlView.onPause();
        }
        
        // Turn off the flash
        if (mFlashOptionView != null && mFlash)
        {
            // OnCheckedChangeListener is called upon changing the checked state
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR1)
            {
                ((Switch) mFlashOptionView).setChecked(false);
            } else
            {
                ((CheckBox) mFlashOptionView).setChecked(false);
            }
        }
        
        try
        {
            vuforiaAppSession.pauseAR();
        } catch (ApplicationException e)
        {
            Log.e(LOGTAG, e.getString());
        }
    }
    
    
    // The final call you receive before your activity is destroyed.
    @Override
    protected void onDestroy()
    {
        Log.d(LOGTAG, "onDestroy");
        super.onDestroy();
        
        try
        {
            vuforiaAppSession.stopAR();
        } catch (ApplicationException e)
        {
            Log.e(LOGTAG, e.getString());
        }
        
        // Unload texture:
        mTextures.clear();
        mTextures = null;
        
        System.gc();
    }
    
    
    // Initializes AR application components.
    private void initApplicationAR()
    {
        // Create OpenGL ES view:
        int depthSize = 16;
        int stencilSize = 0;
        boolean translucent = Vuforia.requiresAlpha();
        
        mGlView = new ApplicationGLView(this);
        mGlView.init(translucent, depthSize, stencilSize);
        
        mRenderer = new ImageTargetRenderer(this, vuforiaAppSession);
        mRenderer.setTextures(mTextures);
        mGlView.setRenderer(mRenderer);
        
    }
    
    
    private void startLoadingAnimation()
    {
        LayoutInflater inflater = LayoutInflater.from(this);
        mUILayout = (RelativeLayout) inflater.inflate(R.layout.camera_overlay,
            null, false);
        
        mOverlayView = (RelativeLayout) inflater.inflate(R.layout.info_overlay, 
        	null);
        
        
        mUILayout.setVisibility(View.VISIBLE);
        mUILayout.setBackgroundColor(Color.BLACK);
        
        // Gets a reference to the loading dialog
        loadingDialogHandler.mLoadingDialogContainer = mUILayout
            .findViewById(R.id.loading_indicator);
        
        // Shows the loading indicator at start
        loadingDialogHandler
            .sendEmptyMessage(LoadingDialogHandler.SHOW_LOADING_DIALOG);
        
        // Adds the inflated layout to the view
        addContentView(mUILayout, new LayoutParams(LayoutParams.MATCH_PARENT,
            LayoutParams.MATCH_PARENT));
        
    }
    
    
    // Methods to load and destroy tracking data.
    @Override
    public boolean doLoadTrackersData()
    {
        TrackerManager tManager = TrackerManager.getInstance();
        ImageTracker imageTracker = (ImageTracker) tManager
            .getTracker(ImageTracker.getClassType());
        if (imageTracker == null)
            return false;
        
        if (mCurrentDataset == null)
            mCurrentDataset = imageTracker.createDataSet();
        
        if (mCurrentDataset == null)
            return false;
        
        if (!mCurrentDataset.load(
            mDatasetStrings.get(mCurrentDatasetSelectionIndex),
            STORAGE_TYPE.STORAGE_APPRESOURCE))
            return false;
        
        if (!imageTracker.activateDataSet(mCurrentDataset))
            return false;
        
        int numTrackables = mCurrentDataset.getNumTrackables();
        for (int count = 0; count < numTrackables; count++)
        {
            Trackable trackable = mCurrentDataset.getTrackable(count);
            if(isExtendedTrackingActive())
            {
                trackable.startExtendedTracking();
            }
            
            String name = "Current Dataset : " + trackable.getName();
            trackable.setUserData(name);
            Log.d(LOGTAG, "UserData:Set the following user data "
                + (String) trackable.getUserData());
        }
        
        return true;
    }
    
    
    @Override
    public boolean doUnloadTrackersData()
    {
        // Indicate if the trackers were unloaded correctly
        boolean result = true;
        
        TrackerManager tManager = TrackerManager.getInstance();
        ImageTracker imageTracker = (ImageTracker) tManager
            .getTracker(ImageTracker.getClassType());
        if (imageTracker == null)
            return false;
        
        if (mCurrentDataset != null && mCurrentDataset.isActive())
        {
            if (imageTracker.getActiveDataSet().equals(mCurrentDataset)
                && !imageTracker.deactivateDataSet(mCurrentDataset))
            {
                result = false;
            } else if (!imageTracker.destroyDataSet(mCurrentDataset))
            {
                result = false;
            }
            
            mCurrentDataset = null;
        }
        
        return result;
    }
    
    
    @SuppressWarnings("deprecation")
	@Override
    public void onInitARDone(ApplicationException exception)
    {
        
        if (exception == null)
        {
            initApplicationAR();
            
            mRenderer.mIsActive = true;
            
            // Now add the GL surface view. It is important
            // that the OpenGL ES surface view gets added
            // BEFORE the camera is started and video
            // background is configured.
            addContentView(mGlView, new LayoutParams(LayoutParams.MATCH_PARENT,
                LayoutParams.MATCH_PARENT));
		    
            addContentView(mOverlayView, new LayoutParams(
                    LayoutParams.FILL_PARENT,
                    LayoutParams.FILL_PARENT));

            // Sets the UILayout to be drawn in front of the camera
            mUILayout.bringToFront();
            
            // Sets the layout background to transparent
            mUILayout.setBackgroundColor(Color.TRANSPARENT);
            
            try
            {
                vuforiaAppSession.startAR(CameraDevice.CAMERA.CAMERA_DEFAULT);
            } catch (ApplicationException e)
            {
                Log.e(LOGTAG, e.getString());
            }
            
            boolean result = CameraDevice.getInstance().setFocusMode(
                CameraDevice.FOCUS_MODE.FOCUS_MODE_CONTINUOUSAUTO);
            
            if (result)
                mContAutofocus = true;
            else
                Log.e(LOGTAG, "Unable to enable continuous autofocus");
            
            mSampleAppMenu = new AppMenu(this, this, "Image Targets",
                mGlView, mUILayout, null);
            setSampleAppMenuSettings();
            
        } else
        {
            Log.e(LOGTAG, exception.getString());
            finish();
        }
    }
    
    
    @Override
    public void onQCARUpdate(State state)
    {
        if (mSwitchDatasetAsap)
        {
            mSwitchDatasetAsap = false;
            TrackerManager tm = TrackerManager.getInstance();
            ImageTracker it = (ImageTracker) tm.getTracker(ImageTracker
                .getClassType());
            if (it == null || mCurrentDataset == null
                || it.getActiveDataSet() == null)
            {
                Log.d(LOGTAG, "Failed to swap datasets");
                return;
            }
            
            doUnloadTrackersData();
            doLoadTrackersData();
        }
    }
    
    
    @Override
    public boolean doInitTrackers()
    {
        // Indicate if the trackers were initialized correctly
        boolean result = true;
        
        TrackerManager tManager = TrackerManager.getInstance();
        Tracker tracker;
        
        // Trying to initialize the image tracker
        tracker = tManager.initTracker(ImageTracker.getClassType());
        if (tracker == null)
        {
            Log.e(
                LOGTAG,
                "Tracker not initialized. Tracker already initialized or the camera is already started");
            result = false;
        } else
        {
            Log.i(LOGTAG, "Tracker successfully initialized");
        }
        return result;
    }
    
    
    @Override
    public boolean doStartTrackers()
    {
        // Indicate if the trackers were started correctly
        boolean result = true;
        
        Tracker imageTracker = TrackerManager.getInstance().getTracker(
            ImageTracker.getClassType());
        if (imageTracker != null)
            imageTracker.start();
        
        return result;
    }
    
    
    @Override
    public boolean doStopTrackers()
    {
        // Indicate if the trackers were stopped correctly
        boolean result = true;
        
        Tracker imageTracker = TrackerManager.getInstance().getTracker(
            ImageTracker.getClassType());
        if (imageTracker != null)
            imageTracker.stop();
        
        return result;
    }
    
    
    @Override
    public boolean doDeinitTrackers()
    {
        // Indicate if the trackers were deinitialized correctly
        boolean result = true;
        
        TrackerManager tManager = TrackerManager.getInstance();
        tManager.deinitTracker(ImageTracker.getClassType());
        
        return result;
    }
    
    
    @Override
    public boolean onTouchEvent(MotionEvent event)
    {
        // Process the Gestures
        if (mSampleAppMenu != null && mSampleAppMenu.processEvent(event))
            return true;
        
        return mGestureDetector.onTouchEvent(event);
    }
    
    
    boolean isExtendedTrackingActive()
    {
        return mExtendedTracking;
    }
    
    final public static int CMD_BACK = -1;
    final public static int CMD_EXTENDED_TRACKING = 1;
    final public static int CMD_AUTOFOCUS = 2;
    final public static int CMD_FLASH = 3;
    final public static int CMD_CAMERA_FRONT = 4;
    final public static int CMD_CAMERA_REAR = 5;
    final public static int CMD_DATASET_START_INDEX = 6;
    
    
    // This method sets the menu's settings
    private void setSampleAppMenuSettings()
    {
        AppMenuGroup group;
        
        group = mSampleAppMenu.addGroup("", false);
        group.addTextItem(getString(R.string.menu_back), -1);
        
        group = mSampleAppMenu.addGroup("", true);
        group.addSelectionItem(getString(R.string.menu_extended_tracking),
            CMD_EXTENDED_TRACKING, false);
        group.addSelectionItem(getString(R.string.menu_contAutofocus),
            CMD_AUTOFOCUS, mContAutofocus);
        mFlashOptionView = group.addSelectionItem(
            getString(R.string.menu_flash), CMD_FLASH, false);
        
        CameraInfo ci = new CameraInfo();
        boolean deviceHasFrontCamera = false;
        boolean deviceHasBackCamera = false;
        for (int i = 0; i < Camera.getNumberOfCameras(); i++)
        {
            Camera.getCameraInfo(i, ci);
            if (ci.facing == CameraInfo.CAMERA_FACING_FRONT)
                deviceHasFrontCamera = true;
            else if (ci.facing == CameraInfo.CAMERA_FACING_BACK)
                deviceHasBackCamera = true;
        }
        
        if (deviceHasBackCamera && deviceHasFrontCamera)
        {
            group = mSampleAppMenu.addGroup(getString(R.string.menu_camera),
                true);
            group.addRadioItem(getString(R.string.menu_camera_front),
                CMD_CAMERA_FRONT, false);
            group.addRadioItem(getString(R.string.menu_camera_back),
                CMD_CAMERA_REAR, true);
        }
        
        group = mSampleAppMenu
            .addGroup(getString(R.string.menu_datasets), true);
        mStartDatasetsIndex = CMD_DATASET_START_INDEX;
        mDatasetsNumber = mDatasetStrings.size();
        
        group.addRadioItem("DLab_marker", mStartDatasetsIndex, true);
     
        mSampleAppMenu.attachMenu();
    }
    
    
    //Update data onscreen
	public void updateByteCount(int[] count, int pixel, float x, float y, boolean bol) {
		
		TextView text_bytecurrent = (TextView) findViewById(R.id.byte_count);
		TextView text_warning = (TextView) findViewById(R.id.warning);
		
	    TextView text_pr = (TextView) findViewById(R.id.pr_count);
		TextView text_ch = (TextView) findViewById(R.id.ch_count);
		TextView text_rr = (TextView) findViewById(R.id.rr_count);
		TextView text_k = (TextView) findViewById(R.id.k_count);
		TextView text_gg = (TextView) findViewById(R.id.gg_count);
		TextView text_fe = (TextView) findViewById(R.id.fe_count);
		
		Display display = getWindowManager().getDefaultDisplay();
		Point size = new Point();
		display.getSize(size);
		
		text_bytecurrent.setTextColor(pixel);
		setCurrentColor(text_bytecurrent.getCurrentTextColor());
		
		text_warning.setX(display.getWidth()/2f);
		text_warning.setY(display.getHeight()*(3f/4f));
		
		text_bytecurrent.setX(0);
		text_bytecurrent.setY(0);
		
		int[] projected = cameraToScreen(x, y);
		
		if(!bol) {
			text_warning.setText("Warning:" + "\n" + "Your cholesterol is too high!");
			
			text_pr.setText("Pr" + "\n" + "250");
			text_ch.setText("CH" + "\n" + "250");
			text_rr.setText("RR" + "\n" + "250");
			text_k.setText("K" + "\n" + "250");
			text_gg.setText("GG" + "\n" + "250");
			text_fe.setText("Fe" + "\n" + "250");

			text_pr.setX(projected[0] + 500);
			text_pr.setY(projected[1] - 50);
			
			text_fe.setX(projected[0] + 300);
			text_fe.setY(projected[1] - 50);

			text_ch.setX(projected[0] + 100);
			text_ch.setY(projected[1] - 50);
			
			text_gg.setX(projected[0] + 400);
			text_gg.setY(projected[1] - 250);
			
			text_k.setX(projected[0] + 200);
			text_k.setY(projected[1] - 250);

			text_rr.setX(projected[0] + 0);
			text_rr.setY(projected[1] - 250); 
			
			return;
		}

	}
	
	private int[] cameraToScreen(float x, float y) {
	    
		VideoMode videoMode = CameraDevice.getInstance().getVideoMode(0);
	    
	    int[] projection = new int[2];
	    
	    Display display = getWindowManager().getDefaultDisplay();
		Point size = new Point();
		display.getSize(size);
		int screenWidth = size.x;
		int screenHeight = size.y;
		int orientation = display.getRotation();
		
	    int xOffset = (int) ((screenWidth - x) / 2.0f + x);
	    int yOffset = (int) ((screenHeight - y) / 2.0f - y);
	    
	    int[] details = mRenderer.getCameraDetails();
	    int configX = details[0];
	    int configY = details[1];
	    
	    // Portrait mode
	    if (orientation == 90 || orientation == 270)
	    {
	        int rotatedX = videoMode.getHeight() - (int) y;
	        int rotatedY = (int) x;
	        
	        projection[0] = (int)(rotatedX * screenWidth / (float) videoMode.getHeight() + xOffset);
	        projection[1] = (int)(rotatedY * screenHeight / (float) videoMode.getWidth() + yOffset);
	    }
	    else
	    {
	        projection[0] = (int)(x * configX / (float) videoMode.getWidth() + xOffset);
	        projection[1] = (int)(y * configY / (float) videoMode.getHeight() + yOffset);
	    }
	    return projection;
	}
    
	public int getCurrentColor() {
		return onScreen;
	}
	
	private void setCurrentColor(int arg) {
		onScreen = arg;
	}
	
    @Override
    public boolean menuProcess(int command)
    {
        
        boolean result = true;
        
        switch (command)
        {
            case CMD_BACK:
                finish();
                break;
            
            case CMD_FLASH:
                result = CameraDevice.getInstance().setFlashTorchMode(!mFlash);
                
                if (result)
                {
                    mFlash = !mFlash;
                } else
                {
                    showToast(getString(mFlash ? R.string.menu_flash_error_off
                        : R.string.menu_flash_error_on));
                    Log.e(LOGTAG,
                        getString(mFlash ? R.string.menu_flash_error_off
                            : R.string.menu_flash_error_on));
                }
                break;
            
            case CMD_AUTOFOCUS:
                
                if (mContAutofocus)
                {
                    result = CameraDevice.getInstance().setFocusMode(
                        CameraDevice.FOCUS_MODE.FOCUS_MODE_NORMAL);
                    
                    if (result)
                    {
                        mContAutofocus = false;
                    } else
                    {
                        showToast(getString(R.string.menu_contAutofocus_error_off));
                        Log.e(LOGTAG,
                            getString(R.string.menu_contAutofocus_error_off));
                    }
                } else
                {
                    result = CameraDevice.getInstance().setFocusMode(
                        CameraDevice.FOCUS_MODE.FOCUS_MODE_CONTINUOUSAUTO);
                    
                    if (result)
                    {
                        mContAutofocus = true;
                    } else
                    {
                        showToast(getString(R.string.menu_contAutofocus_error_on));
                        Log.e(LOGTAG,
                            getString(R.string.menu_contAutofocus_error_on));
                    }
                }
                
                break;
            
            case CMD_CAMERA_FRONT:
            case CMD_CAMERA_REAR:
                
                // Turn off the flash
                if (mFlashOptionView != null && mFlash)
                {
                    // OnCheckedChangeListener is called upon changing the checked state
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR1)
                    {
                        ((Switch) mFlashOptionView).setChecked(false);
                    } else
                    {
                        ((CheckBox) mFlashOptionView).setChecked(false);
                    }
                }
                
                vuforiaAppSession.stopCamera();
                
                try
                {
                    vuforiaAppSession
                        .startAR(command == CMD_CAMERA_FRONT ? CameraDevice.CAMERA.CAMERA_FRONT
                            : CameraDevice.CAMERA.CAMERA_BACK);
                } catch (ApplicationException e)
                {
                    showToast(e.getString());
                    Log.e(LOGTAG, e.getString());
                    result = false;
                }
                doStartTrackers();
                break;
            
            case CMD_EXTENDED_TRACKING:
                for (int tIdx = 0; tIdx < mCurrentDataset.getNumTrackables(); tIdx++)
                {
                    Trackable trackable = mCurrentDataset.getTrackable(tIdx);
                    
                    if (!mExtendedTracking)
                    {
                        if (!trackable.startExtendedTracking())
                        {
                            Log.e(LOGTAG,
                                "Failed to start extended tracking target");
                            result = false;
                        } else
                        {
                            Log.d(LOGTAG,
                                "Successfully started extended tracking target");
                        }
                    } else
                    {
                        if (!trackable.stopExtendedTracking())
                        {
                            Log.e(LOGTAG,
                                "Failed to stop extended tracking target");
                            result = false;
                        } else
                        {
                            Log.d(LOGTAG,
                                "Successfully started extended tracking target");
                        }
                    }
                }
                
                if (result)
                    mExtendedTracking = !mExtendedTracking;         
                break;
            
            default:
                if (command >= mStartDatasetsIndex
                    && command < mStartDatasetsIndex + mDatasetsNumber)
                {
                    mSwitchDatasetAsap = true;
                    mCurrentDatasetSelectionIndex = command
                        - mStartDatasetsIndex;
                }
                break;
        }       
        return result;
    }
    
    
    private void showToast(String text)
    {
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show();
    }

	public static String getAR_Folder() {
		return AR_FOLDER;
	}
}
