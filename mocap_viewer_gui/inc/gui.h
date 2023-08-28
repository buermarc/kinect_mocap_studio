#ifndef GTKMM_EXAMPLE_RANGEWIDGETS_H
#define GTKMM_EXAMPLE_RANGEWIDGETS_H

#include <gtkmm.h>
#include <memory>
#include <string>
#include <filesystem>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#include <future>




class ExampleWindow : public Gtk::Window
{
public:
    ExampleWindow(const Glib::RefPtr<Gtk::Application>& app);
    virtual ~ExampleWindow();

protected:
    //Signal handlers:
    void on_checkbutton_toggled(const int num);
    void on_combo_position();
    void on_combo_changed_depthmode();
    void on_combo_changed_store_settings_mode();
    void on_combo_changed_store_settings_smoothing();
    void on_combo_changed_store_settings_depth_mode();
    void on_combo_changed_store_settings_frames();
    void on_combo_changed_store_settings_color_mode();
    void on_combo_changed_store_settings_in_file();
    void on_combo_changed_store_settings_out_folder();
    void on_combo_changed_store_settings_out_file();
    void on_button_quit();
    void on_button_stop();
    void on_button_run();

    void on_menu_file_files_dialog_in();
    void on_menu_file_files_dialog_out();
    void on_menu_file_quit();
    void on_menu_file_new();
    void on_dialog_response_in(int response_id);
    void on_dialog_response_out(int response_id);

    struct Config {
        std::string mode = "Record";
        double smoothing = 0;
        std::string depth_mode = "NFOV_2X2BINNED";
        std::string frames = "5";
        std::string color_mode = "1080P";
        std::string in_file = "";
        std::string out_folder = std::filesystem::current_path();;
        std::string out_file = "/out";
        bool write = true;
    };
    Config config;


    //Child widgets:
    Gtk::Box m_VBox_Top, m_HBox_Mid, m_Box_Bottom, m_VBox2, m_VBox3;
    Gtk::Box m_HBox_Combo, m_HBox_Combo_1, m_Box_Combo_DepthMode, 
             m_Box_Combo_Frames, m_VBox_Smoothing, m_Box_InFile,
             m_Box_OutOuter, m_Box_OutFolderInner, m_Box_OutFileInner,
             m_Box_TextView;

    Glib::RefPtr<Gtk::Adjustment> m_adjustment, m_adjustment_digits, m_adjustment_pagesize;

    Gtk::Entry m_Entry_InFile;
    Gtk::Label* m_InFile_Label;

    Gtk::Entry m_Entry_OutFolder;
    Gtk::Label* m_OutFolder_Label;
    Gtk::Entry m_Entry_OutFile;

    Gtk::Label* m_Smoothing_Label;
    Gtk::Label* m_Frames_Label;

    Gtk::Separator m_Separator_Bottom;
    Gtk::Separator m_Separator_Mid;

    Gtk::CheckButton m_CheckButton;
    Gtk::CheckButton m_CheckButton_DepthMode;
    Gtk::CheckButton m_CheckButton_OutFile;

    Glib::RefPtr<Gtk::Adjustment> m_adjustment_smoothing;
    Gtk::SpinButton m_Smoothing_SpinButton;

    Glib::RefPtr<Gtk::RecentManager> m_refRecentManager;
    std::unique_ptr<Gtk::FileChooserDialog> m_pDialog_InFile;
    std::unique_ptr<Gtk::FileChooserDialog> m_pDialog_OutFile;

    //Tree model columns:
    class ModelColumns : public Gtk::TreeModel::ColumnRecord
    {
    public:

    ModelColumns()
    { add(m_col_title); add(m_pos); }

    Gtk::TreeModelColumn<Glib::ustring> m_col_title;
    Gtk::TreeModelColumn<guint> m_pos;
    };

    ModelColumns m_Columns_1;
    //Child widgets:
    Gtk::ComboBox m_ComboBox_Mode;
    Glib::RefPtr<Gtk::ListStore> m_refTreeModel_1;

    ModelColumns m_Columns_DepthMode;
    //Child widgets:
    Gtk::ComboBox m_ComboBox_Position_DepthMode;
    Glib::RefPtr<Gtk::ListStore> m_refTreeModel_DepthMode;

    ModelColumns m_Columns;
    //Child widgets:
    Gtk::ComboBox m_ComboBox_ColorMode;
    Glib::RefPtr<Gtk::ListStore> m_refTreeModel;

    ModelColumns m_Columns_Frames;
    //Child widgets:
    Gtk::ComboBox m_ComboBox_Frames;
    Glib::RefPtr<Gtk::ListStore> m_refTreeModel_Frames;

    Gtk::Button m_Button_InFile;
    Gtk::Button m_Button_OutFile;
    Gtk::Button m_Button_Quit;
    Gtk::Button m_Button_Stop;
    Gtk::Button m_Button_Run;

    Gtk::TextView text_view;

    // Shared memory to tell kinect_mocap_studio when to stop
    key_t key;
    int shmid;
    bool* shared_memory = new bool(false);

    std::future<void> result;
};

#endif //GTKMM_EXAMPLE_RANGEWIDGETSH
