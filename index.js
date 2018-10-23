var username=prompt('Enter username')
// $.ajax({
//   type: "GET",
//   url: "http://127.0.0.1:5000/username/"+username,
//   response = HttpResponse(json_data, content_type='application/json'),
//   success: function(data){
//     console.log('User details'+data)
//   }

//   });

// var socket = io.connect('http://127.0.0.1:5000');
// socket.emit('username', username)
var out_l;
var txt=" [More]";
var nxt=" [Next doc match]";
var welcome_text="Hello, how can I help you,"


var $messages = $('.messages-content'),
  d, h, m,
  i=0;
  $(window).load(function() {

    $messages.mCustomScrollbar();
    welcome(username,welcome_text);
});


  // socket.on('imageConversionByServer', function(data) {

  //   var figure=new Image();
  //   figure.src=data

  //   $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ '</div>').append(figure).appendTo($('.mCSB_container')).addClass('new');
  //   console.log(figure)
  // });


function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}

function setDate(){
  d = new Date()
  if (m != d.getMinutes()) {
    m = d.getMinutes();
    $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
  }
}

function insertMessage() {
  msg = $('.message-input').val();
  var flag=0
  $.ajax({
    type: "POST",
    
    
    url: "http://127.0.0.1:5000/input/"+msg+flag,
    data: JSON.stringify("{'username':'" + username+"','query':'"+msg+"'}"),
    dataType: 'json',
    success: function(data){
      $('.message.loading').remove();
      fakeMessage(data)
    },
    error: function (data){
          $('.message.loading').remove();
          fakeMessage(data);
    }
    },
    console.log(msg,flag)
 );
  // socket.emit('input', msg,flag,username)
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
  setDate();
  $('.message-input').val(null);
  $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
  updateScrollbar();
  setTimeout(function() {
  }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
});


function voting(rel_para,rel_loc){

  $('<div class="votestamp"><div class="sprite sprite-fa-thumbs-up-grey" > '+'</div>').appendTo($('.message:last'));

  $(".sprite-fa-thumbs-up-grey").click(function(){

    rel_para[rel_loc]+=1;
    $.ajax({
      type: "POST",
      data: JSON.stringify("{'rel_para':'" + rel_para +"', 'username':'" + username+"', 'name':'" + msg+"'}"),
      dataType: 'json',
      url: "http://127.0.0.1:5000/relevance",

      }),
      console.log('data sent')
    }
    // socket.emit('relevance',rel_para);
    );
}

function welcome(username,welcome_text){
  console.log('in welcome',username)

  $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ "Hello "+ username +", how can I help you?" +'</div>').appendTo($('.mCSB_container')).addClass('new');
  updateScrollbar();
}


function fakeMessage(output1) {
  updateScrollbar();
  if ($('.message-input').val() != '') {
    return false;
  }
  $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));

  out_l=output1.length
  console.log("output length: ",out_l)
  // flag=0
  Fake=JSON.stringify(output1[0].replace(/(\r\n|\n|\r)/gm," "));
  // console.log(out_l)
  console.log(Fake)

  if (out_l>2) {
    Fake2=JSON.stringify(output1[1].replace(/(\r\n|\n|\r)/gm," "));
    Fake3=JSON.stringify(output1[2].replace(/(\r\n|\n|\r)/gm," "));



    }

    setTimeout(() => {

      var alpha=[0,0,0]
      var theta=0;
      console.log('in timeout')
      console.log('fake in timeout:',Fake)
      $('.message.loading').remove();
      if(out_l>2){

        $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake + '<more>'+ txt + '</more></div>').appendTo($('.mCSB_container')).addClass('new');

        theta=0;
        voting(alpha,theta);
        console.log('Voting completed')
        updateScrollbar();
        $("more").click(function(){
          theta=1;
          $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake2 + '<more2>'+ txt + '</more2>' +'</div>').appendTo($('.mCSB_container')).addClass('new');
          voting(alpha,theta);
          updateScrollbar();
          $("more2").click(function(){
            $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake3 + '<next>'+ nxt + '</next>' + '</div>').appendTo($('.mCSB_container')).addClass('new');
            theta=2;
            voting(alpha,theta);
            updateScrollbar();
            $("next").on('click',function(click){
             $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
              console.log('button click')
              var flag=1;
              $.ajax({
                type: "POST",
                url: "http://127.0.0.1:5000/input/"+msg+flag,
                data: JSON.stringify("{'username':'" + username+"','query':'"+msg+"'}"),
                dataType: 'json',
                success: function(data){
                  $('.message.loading').remove();
                  fakeMessage(data)
                },
                error: function(data){
                }
              }),
              updateScrollbar();

                console.log(msg,flag)

              // socket.emit('input', msg,flag,username )
              // console.log('socket emit log, flag:',flag)
              // fakeMessage();

            });
        });
      });

      }

      else if (out_l==1){

        $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake + '<next>' + nxt + '</next>' + '</div>').appendTo($('.mCSB_container')).addClass('new');
        console.log('in else')
        var theta=0;
        var alpha=[0,0,0]
        voting(alpha,theta);
        updateScrollbar();
        $("next").on('click',function(click){
          $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
          var flag=1;
          $.ajax({
            type: "POST",
            url: "http://127.0.0.1:5000/input/"+msg+flag,
            data: JSON.stringify("{'username':'" + username+"','query':'"+msg+"'}"),
            dataType: 'json',
            success: function(data){
              $('.message.loading').remove();
              fakeMessage(data)
            }
          })
          });


      }


      else {


        $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake +'</div>').appendTo($('.mCSB_container')).addClass('new');
        updateScrollbar();
      }
      $('.message.loading').remove();
    },1000)
     updateScrollbar();

}



