#Here we import the tkinter module and everything from the titanic_model.py file
import tkinter
import tkinter as tk
from titanic_model import *
from EDA import *

class Titanic_GUI:
    def __init__(self):

        #Here we create the main window and give it a title.
        self.main_window = tkinter.Tk()
        self.main_window.geometry("1500x1000")
        self.main_window.title("Titanic Survival Predictor:")

        #Here we create frames to group the widgets.
        self.frame_1 = tk.Frame()
        self.frame_2 = tk.Frame()
        self.frame_3 = tk.Frame()
        self.frame_4 = tk.Frame()
        self.frame_5 = tk.Frame()
        self.frame_6 = tk.Frame()
        self.frame_7 = tk.Frame()
        self.frame_8 = tk.Frame()
        self.frame_9 = tk.Frame()
        self.frame_10 = tk.Frame()
        self.frame_11 = tk.Frame()
        self.frame_12 = tk.Frame()
        self.frame_13 = tk.Frame()
        self.frame_14 = tk.Frame()



        #Here we create the widget for the title Frame
        self.title_label = tk.Label(self.frame_1, text="Titanic Survival Predictor:")
        self.title_label.pack()

        # #Here we will display the first 5 rows:
        # text_box = tk.Text(self.main_window, width=150)
        # text_box.insert(tk.END, df.head().to_string())
        # text_box.pack(side="left")

        #Here we create a widget for the PassengerID label.
        self.passenger_id = tk.Label(self.frame_2, text="PassengerID:")
        self.passenger_id.pack(side="left")
        #Here we create the entry widget for the PassengerID.
        self.passenger_id_entry = tk.Entry(self.frame_2)
        self.passenger_id_entry.pack(side="left")

        # Here we create a widget for the Passenger Class label.
        self.pc_class_label = tk.Label(self.frame_3, text="Passenger Class:")
        self.pc_class_label.pack(side="left")
        self.click_pc_class_var = tk.StringVar()
        self.click_pc_class_var.set("Class 1")
        self.pc_class_input = tk.OptionMenu(self.frame_3, self.click_pc_class_var, "Class 1", "Class 2", "Class 3")
        self.pc_class_input.pack(side="left")


        # Here we create a widget for the Name label.
        self.name_label = tk.Label(self.frame_4, text="Name:")
        self.name_label.pack(side="left")
        # Here we create an entry widget for the PC Class
        self.name_entry = tk.Entry(self.frame_4)
        self.name_entry.pack(side="left")

        # Here we create a widget for the Sex label.
        self.sex_label = tk.Label(self.frame_5, text="Sex:")
        self.sex_label.pack(side="left")
        self.sex_var = tk.StringVar()
        self.sex_var.set("Male")
        self.sex_input = tk.OptionMenu(self.frame_5, self.sex_var, "Male", "Female")
        self.sex_input.pack(side="left")

        # Here we create a widget for the Age label.
        self.age_label = tk.Label(self.frame_6, text="Age:")
        self.age_label.pack(side="left")
        # Here we create an entry widget for the PC Class
        self.age_entry = tk.Entry(self.frame_6)
        self.age_entry.pack(side="left")

        # Here we create a widget for the Sibling/Spouse label.
        self.sibling_spouse_label = tk.Label(self.frame_7, text="Sibling/Spouse:")
        self.sibling_spouse_label.pack(side="left")
        # Here we create an entry widget for the PC Class
        self.sibling_spouse_entry = tk.Entry(self.frame_7)
        self.sibling_spouse_entry.pack(side="left")

        # Here we create a widget for the Parent/Children label.
        self.parch_label = tk.Label(self.frame_8, text="Parent/Children:")
        self.parch_label.pack(side="left")
        # Here we create an entry widget for the PC Class
        self.parch_entry = tk.Entry(self.frame_8)
        self.parch_entry.pack(side="left")

        # Here we create a widget for the Ticket label.
        self.ticket_label = tk.Label(self.frame_9, text="Ticket:")
        self.ticket_label.pack(side="left")
        # Here we create an entry widget for the PC Class
        self.ticket_entry = tk.Entry(self.frame_9)
        self.ticket_entry.pack(side="left")

        # Here we create a widget for the Fare label.
        self.fare_label = tk.Label(self.frame_10, text="Fare")
        self.fare_label.pack(side="left")
        # Here we create an entry widget for the PC Class
        self.fare_entry = tk.Entry(self.frame_10)
        self.fare_entry.pack(side="left")

        # Here we create a widget for the Cabin label.
        self.cabin_label = tk.Label(self.frame_11, text="Cabin:")
        self.cabin_label.pack(side="left")
        # Here we create an entry widget for the PC Class
        self.cabin_entry = tk.Entry(self.frame_11)
        self.cabin_entry.pack(side="left")

        # Here we create a widget for the Embarked label.
        self.embarked_label = tk.Label(self.frame_12, text="Embarked:")
        self.embarked_label.pack(side="left")
        self.embarked_var = tk.StringVar()
        self.embarked_var.set("Cherbourg")
        self.embarked_input = tk.OptionMenu(self.frame_12, self.embarked_var, "Cherbourg", "Queenstown", "Southampton")
        self.embarked_input.pack(side="left")

        #Here we create the frame for the prediction of survivability
        self.text_area = tk.Text(self.frame_14, height=12, width=24, bg="light blue")
        self.text_area.pack(side="left")

        #Here we create the prediction and quit buttons
        self.predict_botton = tk.Button(self.frame_13, text="Predict Survivability", command=self.prediction_survived)
        self.predict_botton.pack(side="left")
        self.quit_button = tk.Button(self.frame_13, text="Quit", command=self.main_window.destroy)
        self.quit_button.pack(side="left")




        #Here we unpack all the frames.
        self.frame_1.pack()
        self.frame_2.pack()
        self.frame_3.pack()
        self.frame_4.pack()
        self.frame_5.pack()
        self.frame_6.pack()
        self.frame_7.pack()
        self.frame_8.pack()
        self.frame_9.pack()
        self.frame_10.pack()
        self.frame_11.pack()
        self.frame_12.pack()
        self.frame_13.pack()
        self.frame_14.pack()

        #Here we loop the main window
        tk.mainloop()

    def prediction_survived(self):
        result = ""

        self.text_area.delete(0.0, tk.END)
        #Here we retrieve the passenger ID
        passenger_id = self.passenger_id_entry.get()

        #Here we retrieve the passenger class
        passenger_class = self.click_pc_class_var.get()
        if passenger_class == "1":
            passenger_class = 1
        elif passenger_class == "2":
            passenger_class = 2
        else:
            passenger_class = 3

        #Here we retrieve the passenger's name
        passenger_name = self.passenger_id_entry.get()

        #Here we retrieve the passenger's sex
        passenger_sex = self.sex_var.get()
        if passenger_sex == "Male":
            passenger_sex = 1
        else:
            passenger_sex = 2


        #Here we retrieve the passenger's age
        passenger_age = self.age_entry.get()

        #Here we retrieve the passengers sibling spouse count
        passenger_sib_sp = self.sibling_spouse_entry.get()

        #Here we retrieve the passenger's parent child count
        passenger_par_ch = self.parch_entry.get()

        #Here we retrieve the passenger's ticket number
        passenger_ticket = self.ticket_entry.get()

        #Here we retrieve the passenger's fare
        passenger_fare = self.fare_entry.get()

        #Here we retrieve the passenger's cabin
        passenger_cabin = self.cabin_entry.get()

        #Here we retrieve the passenger's port of embarkation
        passenger_embarked = self.embarked_var.get()
        if passenger_embarked == "Cherbourg":
            passenger_embarked = 1
        elif passenger_embarked == "Queenstown":
            passenger_embarked = 2
        else:
            passenger_embarked = 3

        #Here we collect all the passenger's information together for use in the prediction model.
        result_string = "Prediction of Survivability \n"
        passenger_info = (passenger_id, passenger_class, passenger_name, passenger_sex, passenger_age, passenger_sib_sp,
                          passenger_par_ch, passenger_ticket, passenger_fare, passenger_cabin, passenger_embarked)

        #Here we have model use the passenger's information
        survivability_prediction = best_model.predict([passenger_info])
        display_info = "This prediction has an accuracy of:", str(model_accuracy)

        prediction_result = survivability_prediction

        if survivability_prediction == [0]:
            result_string = str(display_info), "\n", "0 - This passenger didn't survive!"
        else:
            result_string = str(display_info), "\n", "1 - This passenger did survive!"

        self.text_area.insert('1.0', result_string)


myGui = Titanic_GUI()



