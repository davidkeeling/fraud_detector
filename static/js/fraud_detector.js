
var dashboard_data;

function new_prediction(event) {
  event.preventDefault();
  $('body').addClass('wait');
  var parent_el = $('#output');
  parent_el.html('');
  
  $.ajax({
    url: '/score',
    method: 'POST',
    dataType: 'json',
    success: function (data) {
      console.log(data);
      $('body').removeClass('wait');
      render_dashboard_table([data], parent_el);
    },
    error: function (jqXHR, status, err) {
      console.log(jqXHR, status, err);
      $('body').removeClass('wait');
    }
  });
}

function show_tab(tab_id) {
  return function (event) {
    if (event) {
      event.preventDefault();
    }
    $('.tab_content').addClass('hidden');
    $('#' + tab_id).removeClass('hidden');
  }
}

function render_dashboard_table(data, parent_el) {
  parent_el.html('');

  var headers = ['text_desc', 'fraud_predicted', 'fraud_probability'];
  var table = $('<table></table>');
  var table_header_row = $('<tr></tr>');
  for (var h in headers) {
    table_header_row.append($('<th>' + headers[h] + '</th>'));
  }
  table.append(table_header_row);
  for (var i = 0; i < data.length; i++) {
    var row = $('<tr></tr>')
    for (var h in headers) {
      row.append($('<td><div>' + data[i][headers[h]] + '</div></td>'));
    }
    table.append(row);
  }
  parent_el.append(table);
}

$(document).ready(function () {

  $('#score').on('submit', new_prediction);

  $('#home_btn').on('click', show_tab('home_content'));
  $('#score_btn').on('click', show_tab('score_content'));

  $('#dashboard_btn').on('click', function (event) {
    event.preventDefault();
    var parent_el = $('#dashboard_content');

    if (dashboard_data) {
      console.log('using loaded dashboard data');
      render_dashboard_table(dashboard_data, parent_el);
      show_tab('dashboard_content')();
      return;
    }

    $('body').addClass('wait');
    console.log('fetching dashboard data...');
    $.ajax({
      url: '/dashboard',
      method: 'GET',
      dataType: 'json',
      success: function (data) {
        console.log(data);
        dashboard_data = data;
        render_dashboard_table(dashboard_data, parent_el);
        $('body').removeClass('wait');
        show_tab('dashboard_content')();
      },
      error: function (jqXHR, status, err) {
        console.log(jqXHR, status, err);
        $('body').removeClass('wait');
      }
    })
  });

  $('.nav_tab_btn').on('click', function (event) {
    event.preventDefault();
    $('.nav_tab_btn').removeClass('active');
    $(event.target).closest('.nav_tab_btn').addClass('active');
  });
});
