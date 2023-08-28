#include "gui.h"
#include <iostream>
#include <algorithm>
#include "kinect_mocap_studio.cc"
#include <future>





ExampleWindow::ExampleWindow(const Glib::RefPtr<Gtk::Application>& app)
:
    m_refRecentManager(Gtk::RecentManager::get_default()),
    m_VBox_Top(Gtk::Orientation::VERTICAL, 0),
    m_HBox_Mid(Gtk::Orientation::HORIZONTAL, 20),
    m_Box_Bottom(Gtk::Orientation::HORIZONTAL, 5),
    m_VBox2(Gtk::Orientation::VERTICAL, 20),
    m_VBox3(Gtk::Orientation::VERTICAL, 20),
    m_HBox_Combo(Gtk::Orientation::VERTICAL, 5),
    m_Box_Combo_DepthMode(Gtk::Orientation::VERTICAL, 5),
    m_Box_Combo_Frames(Gtk::Orientation::VERTICAL, 5),
    m_HBox_Combo_1(Gtk::Orientation::HORIZONTAL, 10),
    m_Box_InFile(Gtk::Orientation::HORIZONTAL, 5),
    m_Box_OutFolderInner(Gtk::Orientation::HORIZONTAL, 5),
    m_Box_OutFileInner(Gtk::Orientation::HORIZONTAL, 5),
    m_Box_OutOuter(Gtk::Orientation::VERTICAL, 5),
    m_VBox_Smoothing(Gtk::Orientation::VERTICAL, 5),
    m_Box_TextView(Gtk::Orientation::VERTICAL, 20),

  // A checkbutton to control whether the value is displayed or not:
    m_adjustment_smoothing(Gtk::Adjustment::create(0.0, 0.0, 1.0, 0.1, 0.1, 0)),
    m_Smoothing_SpinButton(m_adjustment_smoothing, 1.0, 2),
    m_CheckButton("Color mode:", 0),
    m_CheckButton_DepthMode("Depth mode:", 0),
    m_CheckButton_OutFile("Save:", 0),

    m_Button_InFile("Choose File"),
    m_Button_OutFile("Choose Folder"),
    m_Button_Quit("Quit"),
    m_Button_Stop("Stop"),
    m_Button_Run("Run")
{
    set_title("Mocap Studio");
    set_default_size(800, 600);

    set_child(m_VBox_Top);
    m_VBox_Top.append(m_HBox_Mid);

//     m_HBox_Mid.set_homogeneous(true);

    m_HBox_Mid.append(m_VBox2);
    m_VBox2.set_expand(false);
    m_VBox2.set_margin(10);

    m_HBox_Mid.append(m_Separator_Mid);

    m_HBox_Mid.append(m_VBox3);
    m_VBox3.set_expand(true);
    m_VBox3.set_margin(10);

    //////////////////////////////////////////////////
    /// Smoothing
    //////////////////////////////////////////////////
    m_Smoothing_SpinButton.set_wrap();
    m_Smoothing_SpinButton.set_size_request(100, -1);
    m_Smoothing_SpinButton.set_numeric(true);

    m_Smoothing_Label = Gtk::make_managed<Gtk::Label>("Smoothing:", 0);
    m_Smoothing_Label->set_xalign(0.0);

    //ExampleWindow:config.smoothing = 0.0;
    m_Smoothing_SpinButton.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_smoothing));

    m_VBox_Smoothing.append(*m_Smoothing_Label);
    m_VBox_Smoothing.append(m_Smoothing_SpinButton);
    m_VBox2.append(m_VBox_Smoothing);

    //////////////////////////////////////////////////
    /// Depth mode
    //////////////////////////////////////////////////
    //CheckButton:
    m_CheckButton_DepthMode.set_active();
    m_Box_Combo_DepthMode.append(m_CheckButton_DepthMode);

    //Position ComboBox:
    //Create the Tree model:
    m_refTreeModel_DepthMode = Gtk::ListStore::create(m_Columns_DepthMode);
    m_ComboBox_Position_DepthMode.set_model(m_refTreeModel_DepthMode);
    m_ComboBox_Position_DepthMode.pack_start(m_Columns_DepthMode.m_col_title);

    //Fill the ComboBox's Tree Model:
    auto row_depthmode = *(m_refTreeModel_DepthMode->append());
    row_depthmode[m_Columns_DepthMode.m_col_title] = "NFOV_2x2BINNED";

    row_depthmode = *(m_refTreeModel_DepthMode->append());
    row_depthmode[m_Columns_DepthMode.m_col_title] = "NFOV_UNBINNED";

    row_depthmode = *(m_refTreeModel_DepthMode->append());
    row_depthmode[m_Columns_DepthMode.m_col_title] = "WFOV_2x2BINNED";

    row_depthmode = *(m_refTreeModel_DepthMode->append());
    row_depthmode[m_Columns_DepthMode.m_col_title] = "WFOV_UNBINNED";

    row_depthmode = *(m_refTreeModel_DepthMode->append());
    row_depthmode[m_Columns_DepthMode.m_col_title] = "PASSIVE_IR";

    m_CheckButton_DepthMode.signal_toggled().connect(sigc::bind(sigc::mem_fun(*this,
                                                                              &ExampleWindow::on_checkbutton_toggled), 
                                                                0));
    m_ComboBox_Position_DepthMode.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_depthmode));
    m_ComboBox_Position_DepthMode.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_depth_mode));

    m_VBox2.append(m_Box_Combo_DepthMode);
    m_Box_Combo_DepthMode.append(m_ComboBox_Position_DepthMode);
    m_ComboBox_Position_DepthMode.set_active(0); // Top
    m_ComboBox_Position_DepthMode.set_expand(false);

    //////////////////////////////////////////////////
    /// Color mode
    //////////////////////////////////////////////////
    //CheckButton:
    m_CheckButton.set_active();
    m_HBox_Combo.append(m_CheckButton);

    //Create the Tree model:
    m_refTreeModel = Gtk::ListStore::create(m_Columns);
    m_ComboBox_ColorMode.set_model(m_refTreeModel);
    m_ComboBox_ColorMode.pack_start(m_Columns.m_col_title);

    //Fill the ComboBox's Tree Model:
    auto row = *(m_refTreeModel->append());
    row[m_Columns.m_col_title] = "720P";

    row = *(m_refTreeModel->append());
    row[m_Columns.m_col_title] = "1080P";

    row = *(m_refTreeModel->append());
    row[m_Columns.m_col_title] = "1440P";

    row = *(m_refTreeModel->append());
    row[m_Columns.m_col_title] = "1536P";

    row = *(m_refTreeModel->append());
    row[m_Columns.m_col_title] = "2160P";

    row = *(m_refTreeModel->append());
    row[m_Columns.m_col_title] = "3072P";

    m_CheckButton.signal_toggled().connect(sigc::bind(sigc::mem_fun(*this,
                                                                    &ExampleWindow::on_checkbutton_toggled), 
                                                      1));
    m_ComboBox_ColorMode.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_color_mode));

    m_VBox2.append(m_HBox_Combo);
    m_HBox_Combo.append(m_ComboBox_ColorMode);
    m_ComboBox_ColorMode.set_active(0); // Top
    m_ComboBox_ColorMode.set_expand(false);

    //////////////////////////////////////////////////
    /// Frames per second
    //////////////////////////////////////////////////
    m_Frames_Label = Gtk::make_managed<Gtk::Label>("Frames/sec:", 0);
    m_Frames_Label->set_xalign(0.0);
    m_Box_Combo_Frames.append(*m_Frames_Label);

    //Create the Tree model:
    m_refTreeModel_Frames = Gtk::ListStore::create(m_Columns_Frames);
    m_ComboBox_Frames.set_model(m_refTreeModel_Frames);
    m_ComboBox_Frames.pack_start(m_Columns_Frames.m_col_title);

    //Fill the ComboBox's Tree Model:
    auto row_frames = *(m_refTreeModel_Frames->append());
    row_frames[m_Columns_Frames.m_col_title] = "5";

    row_frames = *(m_refTreeModel_Frames->append());
    row_frames[m_Columns_Frames.m_col_title] = "15";

    row_frames = *(m_refTreeModel_Frames->append());
    row_frames[m_Columns_Frames.m_col_title] = "30";

    m_ComboBox_Frames.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_frames));

    m_Box_Combo_Frames.append(m_ComboBox_Frames);
    m_ComboBox_Frames.set_active(0); // Top
    m_ComboBox_Frames.set_expand(false);
    m_VBox2.append(m_Box_Combo_Frames);


    //////////////////////////////////////////////////
    /// Mode
    //////////////////////////////////////////////////
    m_refTreeModel_1 = Gtk::ListStore::create(m_Columns_1);
    m_ComboBox_Mode.set_model(m_refTreeModel_1);
    m_ComboBox_Mode.pack_start(m_Columns_1.m_col_title);

    //Fill the ComboBox's Tree Model:
    auto row_1 = *(m_refTreeModel_1->append());
    row_1[m_Columns_1.m_col_title] = "Record";
    row_1[m_Columns_1.m_pos] = 0;
    row_1 = *(m_refTreeModel_1->append());

    row_1[m_Columns_1.m_col_title] = "Reprocess";
    row_1[m_Columns_1.m_pos] = 1;
    m_ComboBox_Mode.set_active(0); // Top
    m_ComboBox_Mode.set_expand(false);

    m_ComboBox_Mode.signal_changed().connect( sigc::mem_fun(*this, &ExampleWindow::on_combo_position));
    m_ComboBox_Mode.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_mode));

    m_HBox_Combo_1.append(*Gtk::make_managed<Gtk::Label>("Mode:", 0));
    m_HBox_Combo_1.append(m_ComboBox_Mode);
    m_VBox3.append(m_HBox_Combo_1);



    //////////////////////////////////////////////////
    /// Console output display
    //////////////////////////////////////////////////
    m_Box_TextView.append(ExampleWindow::text_view);
    m_VBox3.append(m_Box_TextView);
    auto text_buffer = text_view.get_buffer();

    // Set initial text
    text_buffer->set_text("INFO:");

    //////////////////////////////////////////////////
    /// In file
    //////////////////////////////////////////////////
    m_InFile_Label = Gtk::make_managed<Gtk::Label>("In:", 0);
    m_InFile_Label->set_xalign(0.0);
    m_Box_InFile.append(*m_InFile_Label);

    m_Entry_InFile.set_max_length(200);
    m_Entry_InFile.set_text(ExampleWindow::config.in_file);
    m_Entry_InFile.select_region(0, m_Entry_InFile.get_text_length());
    m_Entry_InFile.set_expand(false);
    m_Box_InFile.append(m_Entry_InFile);

    m_Button_InFile.signal_clicked().connect(sigc::mem_fun(*this,
                                                           &ExampleWindow::on_menu_file_files_dialog_in));
    m_Entry_InFile.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_in_file));

    m_Box_InFile.append(m_Button_InFile);
    set_default_widget(m_Button_InFile);
    m_Box_InFile.set_sensitive(false);

    m_VBox2.append(m_Box_InFile);

    //////////////////////////////////////////////////
    /// Out file
    //////////////////////////////////////////////////
    //CheckButton:
    m_CheckButton_OutFile.set_active();
    m_CheckButton_OutFile.signal_toggled().connect(sigc::bind(sigc::mem_fun(*this,
                                                                            &ExampleWindow::on_checkbutton_toggled), 
                                                              2));
    m_Box_OutOuter.append(m_CheckButton_OutFile);

    m_OutFolder_Label = Gtk::make_managed<Gtk::Label>("Out:", 0);
    m_OutFolder_Label->set_xalign(0.0);
    m_Box_OutOuter.append(*m_OutFolder_Label);

    m_Entry_OutFolder.set_max_length(50);
    m_Entry_OutFolder.set_text(ExampleWindow::config.out_folder);
    m_Entry_OutFolder.select_region(0, m_Entry_OutFolder.get_text_length());
    m_Entry_OutFolder.set_expand(false);
    m_Box_OutFolderInner.append(m_Entry_OutFolder);

    set_default_widget(m_Button_OutFile);
    m_Box_OutFolderInner.append(m_Button_OutFile);

    m_Entry_OutFile.set_max_length(50);
    m_Entry_OutFile.set_text(ExampleWindow::config.out_file);
    m_Entry_OutFile.select_region(0, m_Entry_OutFile.get_text_length());
    m_Entry_OutFile.set_expand(false);
    m_Box_OutFileInner.append(m_Entry_OutFile);

    m_Button_OutFile.signal_clicked().connect(sigc::mem_fun(*this,
                                                            &ExampleWindow::on_menu_file_files_dialog_out));
    m_Entry_OutFolder.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_out_folder));
    m_Entry_OutFile.signal_changed().connect(sigc::mem_fun(*this, &ExampleWindow::on_combo_changed_store_settings_out_file));

    m_Box_OutOuter.append(m_Box_OutFolderInner);
    m_Box_OutOuter.append(m_Box_OutFileInner);
    m_VBox2.append(m_Box_OutOuter);

    //////////////////////////////////////////////////
    /// Bottom
    //////////////////////////////////////////////////

    // QUIT
    m_VBox_Top.append(m_Separator_Bottom);

    set_default_widget(m_Button_Quit);
    m_Button_Quit.signal_clicked().connect(sigc::mem_fun(*this,
                                                         &ExampleWindow::on_button_quit));
    m_Button_Quit.set_margin(10);
    m_Box_Bottom.append(m_Button_Quit);

    // STOP
    set_default_widget(m_Button_Stop);
    m_Button_Stop.signal_clicked().connect(sigc::mem_fun(*this,
                                                         &ExampleWindow::on_button_stop));

    m_Button_Stop.set_margin(10);
    m_Box_Bottom.append(m_Button_Stop);

    // RUN
    set_default_widget(m_Button_Run);
    m_Button_Run.signal_clicked().connect(sigc::mem_fun(*this,
                                                         &ExampleWindow::on_button_run));
    m_Button_Run.set_margin(10);
    m_Box_Bottom.append(m_Button_Run);

    m_Box_Bottom.set_halign(Gtk::Align::END);
    m_VBox_Top.append(m_Box_Bottom);

}





ExampleWindow::~ExampleWindow()
{
}

void ExampleWindow::on_checkbutton_toggled(const int num)
{
    switch (num) {
        case 0:
            if (m_ComboBox_Position_DepthMode.get_sensitive()) {
                m_ComboBox_Position_DepthMode.set_sensitive(false);
            } else {
                m_ComboBox_Position_DepthMode.set_sensitive(true);
            }
            break;

        case 1:
            if (m_ComboBox_ColorMode.get_sensitive()) {
                m_ComboBox_ColorMode.set_sensitive(false);
            } else {
                m_ComboBox_ColorMode.set_sensitive(true);
            }
            break;
        case 2:
            if (m_Box_OutFolderInner.get_sensitive() &&
                m_Box_OutFileInner.get_sensitive()) {
                m_Box_OutFolderInner.set_sensitive(false);
                m_Box_OutFileInner.set_sensitive(false);
                ExampleWindow::config.write = false;
            } else {
                m_Box_OutFolderInner.set_sensitive(true);
                m_Box_OutFileInner.set_sensitive(true);
                ExampleWindow::config.write = true;
            }
            break;
    }

}

void ExampleWindow::on_combo_position()
{
    const auto iter = m_ComboBox_Mode.get_active();
    if(!iter)
        return;

    const auto row = *iter;
    if(!row)
        return;

    const auto pos = row[m_Columns_1.m_pos];

    // Record
    if (pos == 0) {
        m_Box_InFile.set_sensitive(false);
        m_CheckButton_OutFile.set_sensitive(true);
        m_CheckButton_OutFile.set_active(true);
    }

    // Reprocess
    if (pos == 1) {
        m_Box_InFile.set_sensitive(true);
        m_CheckButton_OutFile.set_active(false);
        m_CheckButton_OutFile.set_sensitive(false);
    }
}

void ExampleWindow::on_combo_changed_depthmode() {
    auto num = m_ComboBox_Position_DepthMode.get_active_row_number();

    if (m_refTreeModel_Frames != NULL) {
        auto it = m_refTreeModel_Frames->children()[2].get_iter();

        if (it && num == 3) {
            m_refTreeModel_Frames->erase(it);
        }
        
        if (m_refTreeModel_Frames->children().size() == 2 && num != 3) {
            auto row_frames = *(m_refTreeModel_Frames->append());
            row_frames[m_Columns_Frames.m_col_title] = "30";
        }

    }
}

void ExampleWindow::on_combo_changed_store_settings_mode() {
    auto num = m_ComboBox_Mode.get_active_row_number();

    switch(num) {
        case 0:
            ExampleWindow::config.mode = "Record";
            break;
        case 1:
            ExampleWindow::config.mode = "Reprocess";
            break;
    }
}

void ExampleWindow::on_combo_changed_store_settings_smoothing() {
    ExampleWindow::config.smoothing = m_Smoothing_SpinButton.get_value();
}

void ExampleWindow::on_combo_changed_store_settings_depth_mode() {
    auto num = m_ComboBox_Position_DepthMode.get_active_row_number();

    switch(num) {
        case 0:
            ExampleWindow::config.depth_mode = "NFOV_2X2BINNED";
            break;
        case 1:
            ExampleWindow::config.depth_mode = "NFOV_UNBINNED";
            break;
        case 2:
            ExampleWindow::config.depth_mode = "WFOV_2X2BINNED";
            break;
        case 3:
            ExampleWindow::config.depth_mode = "WFOV_UNBINNED";
            break;
        case 4:
            ExampleWindow::config.depth_mode = "PASSIVE_IR";
            break;
    }
}

void ExampleWindow::on_combo_changed_store_settings_frames() {
    auto num = m_ComboBox_Frames.get_active_row_number();

    switch(num) {
        case 0:
            ExampleWindow::config.frames = "5";
            break;
        case 1:
            ExampleWindow::config.frames = "15";
            break;
        case 2:
            ExampleWindow::config.frames = "30";
            break;
    }
}

void ExampleWindow::on_combo_changed_store_settings_color_mode() {
    auto num = m_ComboBox_ColorMode.get_active_row_number();

    switch(num) {
        case 0:
            ExampleWindow::config.color_mode = "720P";
            break;
        case 1:
            ExampleWindow::config.color_mode = "1080P";
            break;
        case 2:
            ExampleWindow::config.color_mode = "1440P";
            break;
        case 3:
            ExampleWindow::config.color_mode = "1536P";
            break;
        case 4:
            ExampleWindow::config.color_mode = "2160P";
            break;
        case 5:
            ExampleWindow::config.color_mode = "3072P";
            break;
    }
}

void ExampleWindow::on_combo_changed_store_settings_in_file() {
    ExampleWindow::config.in_file = m_Entry_InFile.get_text();
}

void ExampleWindow::on_combo_changed_store_settings_out_folder() {
    ExampleWindow::config.out_folder = m_Entry_OutFolder.get_text();
}

void ExampleWindow::on_combo_changed_store_settings_out_file() {
    ExampleWindow::config.out_file = m_Entry_OutFile.get_text();
}

void ExampleWindow::on_button_quit()
{

    if (*shared_memory) {
        *ExampleWindow::shared_memory = false;
        ExampleWindow::result.get();
    }


    // Detach the shared memory segment
    shmdt(ExampleWindow::shared_memory);

    hide();
}

void ExampleWindow::on_button_stop()
{
    if (*shared_memory) {
        *ExampleWindow::shared_memory = false;

        ExampleWindow::result.get();
    }
}

void ExampleWindow::on_button_run()
{
    std::string info = "INFO:\n";
    info += "Mode: " + ExampleWindow::config.mode + "\n";
    info += "Smoothing: " + std::to_string(ExampleWindow::config.smoothing) + "\n";
    info += "Depth mode: " + ExampleWindow::config.depth_mode + "\n";
    info += "Frames: " + ExampleWindow::config.frames + "\n";
    info += "Color mode: " + ExampleWindow::config.color_mode + "\n";
    info += "In file: " + ExampleWindow::config.out_file + "\n";
    info += "Out folder: " + ExampleWindow::config.out_folder + "\n";
    info += "Out file: " + ExampleWindow::config.out_file + "\n";
    info += "Write: " + std::to_string(ExampleWindow::config.write)  + "\n";

    auto text_buffer = ExampleWindow::text_view.get_buffer();
    text_buffer->set_text(info);


    ExampleWindow::key = ftok("shmfile",65);
    ExampleWindow::shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    // Attach the shared memory segment to the process's address space
    ExampleWindow::shared_memory = (bool*)shmat(shmid, NULL, 0);

    if (ExampleWindow::shmid == -1) {
        perror("shmget");
        return;
    }
    if ((int*)ExampleWindow::shared_memory == (int*)(-1)) {
        perror("shmat");
        return;
    }
    *ExampleWindow::shared_memory = true;
    std::cout << *shared_memory << std::endl;
    ExampleWindow::result = std::async(std::launch::async, mocap_studio, ExampleWindow::config.smoothing,
                                                                            ExampleWindow::config.mode == "Record" ? "" : ExampleWindow::config.in_file,
                                                                            std::stoi(ExampleWindow::config.frames),
                                                                            ExampleWindow::config.color_mode,
                                                                            ExampleWindow::config.depth_mode,
                                                                            ExampleWindow::config.mode == "Reprocess" ? true : false,
                                                                            ExampleWindow::config.mode == "Record" ? true : false,
                                                                            ExampleWindow::config.out_folder + ExampleWindow::config.out_file);
}

void ExampleWindow::on_menu_file_new()
{
  std::cout << " New File" << std::endl;
}

void ExampleWindow::on_menu_file_quit()
{
    hide(); //Closes the main window to stop the app->make_window_and_run().
}

void ExampleWindow::on_menu_file_files_dialog_in() {
    if (!m_pDialog_InFile) {
        m_pDialog_InFile.reset(new Gtk::FileChooserDialog(*this, "Files",
        Gtk::FileChooser::Action::OPEN, /* use_header_bar= */ true));
        m_pDialog_InFile->set_transient_for(*this);
        m_pDialog_InFile->set_modal(true);
        m_pDialog_InFile->signal_response().connect(sigc::mem_fun(*this, 
                                                                  &ExampleWindow::on_dialog_response_in));

        m_pDialog_InFile->add_button("Select Folder", Gtk::ResponseType::OK);
        m_pDialog_InFile->add_button("_Cancel", Gtk::ResponseType::CANCEL);
    }
    m_pDialog_InFile->show();
}

void ExampleWindow::on_menu_file_files_dialog_out() {
    if (!m_pDialog_OutFile) {
        m_pDialog_OutFile.reset(new Gtk::FileChooserDialog(*this, "Files",
        Gtk::FileChooser::Action::SELECT_FOLDER, /* use_header_bar= */ true));
        m_pDialog_OutFile->set_transient_for(*this);
        m_pDialog_OutFile->set_modal(true);
        m_pDialog_OutFile->signal_response().connect(sigc::mem_fun(*this, 
                                                                   &ExampleWindow::on_dialog_response_out));

        m_pDialog_OutFile->add_button("Select Folder", Gtk::ResponseType::OK);
        m_pDialog_OutFile->add_button("_Cancel", Gtk::ResponseType::CANCEL);
    }
    m_pDialog_OutFile->show();
}

void ExampleWindow::on_dialog_response_in(int response_id) {
    m_pDialog_InFile->hide();

    if (response_id == Gtk::ResponseType::OK) {
        auto selected_infile = m_pDialog_InFile->get_file()->get_path();
        m_Entry_InFile.set_text(selected_infile);
    }
}

void ExampleWindow::on_dialog_response_out(int response_id) {
    m_pDialog_OutFile->hide();

    if (response_id == Gtk::ResponseType::OK) {
        auto selected_uri_outfile = m_pDialog_OutFile->get_file()->get_uri();
        m_Entry_OutFolder.set_text(selected_uri_outfile);
    }
}
