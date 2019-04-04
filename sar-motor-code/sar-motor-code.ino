#include <SPI.h>
#include <Ethernet.h>
#include <string.h>

#define DUTY_CYCLE 0.6
#define LEFT LOW
#define RIGHT HIGH
#define STEPS_PER_TURN 20000

/**************************GPIO definitions**************************/

int motor_step = 9;
int motor_enable = 8;
int motor_direction = 7;
int switch_left = 5;
int switch_right = 6;
int vna_power = 2;
int pa_power = 3;
int motor_power = 4;

/********************************************************************/

/**************************Global variables**************************/

long steps_moved = 0;
long steps_max = 0;
long steps_global = 0;
int steps_freq = 3000;
int servo_pos = 0;
String instruction = "";
char instruction_character = '\0';
static volatile bool timer_flag = false;
static volatile bool calibration_flag = false;

/********************************************************************/

/**************************TCP configuration*************************/

byte mac[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED};
byte ip[] = {10, 10, 40, 245};
byte gateway[] = {10, 10, 40, 1};
byte subnet[] = {255, 255, 255, 0};
EthernetServer server = EthernetServer(12345);

/********************************************************************/

void setup()
{
  pinMode(motor_step, OUTPUT);
  pinMode(motor_direction, OUTPUT);
  pinMode(motor_enable, OUTPUT);
  pinMode(vna_power, OUTPUT);
  pinMode(pa_power, OUTPUT);
  pinMode(motor_power, OUTPUT);
  pinMode(switch_left, INPUT);
  pinMode(switch_right, INPUT);

  digitalWrite(motor_enable, LOW);
  digitalWrite(vna_power, LOW);
  digitalWrite(pa_power, LOW);
  digitalWrite(motor_power, LOW);

  Ethernet.begin(mac, ip, gateway, subnet);  
  server.begin();
}

void loop()
{
  EthernetClient client = server.available();
  
  if (client)  
    listen_instruction(client);  
}

void listen_instruction(EthernetClient client)
{
  while (client.connected())
  { 
    while (1)
    {
      instruction="";
      char header="";      
      read_client(client);
      header = instruction.charAt(0);      

      //move motor
      if (header == '0')
      {        
        char dir = (instruction.substring(instruction.length() - 1, instruction.length() - 2)).charAt(0);
        steps_global = instruction.substring(1, instruction.length() - 2).toInt();        
        calibration_flag = false;

        if (dir == 'L')
        {          
          //move_motor(steps, LEFT, client);
          timer_flag=false;
          move_motor(LEFT, client);          
          server.println("OK\n");
        }

        if (dir == 'R')
        {
          //move_motor(steps, RIGHT, client);
          timer_flag=false;
          move_motor(RIGHT, client);
          server.println("OK\n");
        }
      }

      //calibrate motor, store number of steps from side to side
      if (header == '1')
      {        
        calibrate(client);
        server.print(String(steps_max) + '\n');
      }

      //go to zero position on the rail
      if (header == '2')
      {
        calibration_flag = true;
        zero_position(client);
        server.println("OK\n");
        calibration_flag = false;        
      }

      //end connection
      if (header == '4')
      {
        server.println("Bye\n");
        break;
      }

      if (header == '5')
      {
        char state = (instruction.substring(instruction.length() - 1, instruction.length() - 2)).charAt(0);

        if (state == '0')
        {
          digitalWrite(vna_power, LOW);
          server.println("VNA OFF\n");
        }

        if (state == '1')
        {
          digitalWrite(vna_power, HIGH);
          server.println("VNA ON\n");
        }
      }

      if (header == '6')
      {        
        char state = (instruction.substring(instruction.length() - 1, instruction.length() - 2)).charAt(0);

        if (state == '0')
        {
          digitalWrite(pa_power, LOW);
          server.println("PA OFF\n");
        }

        if (state == '1')
        {
          digitalWrite(pa_power, HIGH);
          server.println("PA ON\n");
        }
      }
    }
  }  
}

void move_motor(bool dir, EthernetClient client)
{
  instruction = "";
  steps_moved = 0;  
  digitalWrite(motor_direction, dir);
  timer_on();

  while (1)
  {  
    if (timer_flag)
    { 
      break;
    }
  }
}

void calibrate(EthernetClient client)
{
  bool stop_flag = LOW;
  char header = '\0';
  instruction = "";
  timer_on();
  digitalWrite(motor_direction, RIGHT);

  while (1)
  {
    read_client(client);
    header = instruction.charAt(0);

    if (header == '3')
    {
      timer_off();
      stop_flag = HIGH;
      break;
    }

    if (digitalRead(switch_right) == HIGH)
    {
      timer_off();
      digitalWrite(motor_direction, LEFT);
      timer_on();

      while (1)
      {
        if (digitalRead(switch_right) == LOW)
          break;
      }

      timer_off();
      break;
    }
  }

  if (!stop_flag)
  {
    header = '\0';
    steps_moved = 0;
    timer_on();
    digitalWrite(motor_direction, LEFT);

    while (1)
    {
      read_client(client);
      header = instruction.charAt(0);

      if (header == '3')
      {
        timer_off();
        break;
      }

      if (digitalRead(switch_left) == LOW)
      {
        timer_off();
        digitalWrite(motor_direction, RIGHT);
        timer_on();

        while (1)
        {
          if (digitalRead(switch_left) == HIGH)
            break;
        }

        timer_off();
        steps_max = steps_moved;
        break;
      }
    }
  }
  return;
}

void zero_position(EthernetClient client)
{
  instruction="";
  timer_on();
  digitalWrite(motor_direction, LEFT);
  
  while (1)
  { 
    /*    
    read_client(client);
    char header = instruction.charAt(0);    
    
    if (header == '3')
    {
      timer_off();
      break;
    }
    */
        
    if (digitalRead(switch_left) == HIGH)
    {
      timer_off();      
      digitalWrite(motor_direction, RIGHT);
      timer_on();

      while (1)
      {
        if (digitalRead(switch_left) == LOW)
          break;
      }
      timer_off();      
      break;
    }    
  }
}

void read_client(EthernetClient client)
{
  if (client)
  {
    while (client.connected())
    {
      if (client.available())              
        instruction = instruction + char(client.read());
      
      if (instruction.charAt(instruction.length() - 1) == '\n')
        break;
    }
  }
  return;
}

void timer_on()
{
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCCR1A |= (1 << COM1A1 | 1 << COM1A0 | 1 << WGM11);
  TCCR1B |= (1 << WGM12 | 1 << WGM13);
  OCR1A = (1 - DUTY_CYCLE) * steps_freq;
  ICR1 = steps_freq;
  TCCR1B |= (1 << CS10);
  
  if (!calibration_flag)
  {      
    TIMSK1 |= (1 << OCIE1A);
    interrupts();
  }  
}

void timer_off()
{
  TCCR1A = 0;
  TCCR1B = 0;
  TIMSK1 &= (0 << OCIE1A);  
  noInterrupts();  
}

ISR(TIMER1_COMPA_vect)
{ 
  if (steps_moved == steps_global)
  {
    timer_flag = true;
    timer_off();    
  }  
  else
  {
    steps_moved++;
  }
}
